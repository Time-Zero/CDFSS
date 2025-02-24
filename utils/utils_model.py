import math
import os.path
import random
import sys
from functools import partial

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from colorama import Fore, Style
from torch import nn
from torch.amp import autocast
from tqdm import tqdm

from utils.utils_predict import f_score
from joblib import Parallel, delayed


def process_single_file(filename, mask_dir, num_classes):
    """处理单个文件（无分块）"""
    try:
        mask_path = os.path.join(mask_dir, filename)
        with Image.open(mask_path) as img:
            # 转换为灰度并验证
            mask = np.array(img)

            # 检查像素范围
            if np.max(mask) >= num_classes:
                raise ValueError(f"文件 {filename} 包含非法像素值 {np.max(mask)}")

            # 统计像素
            counts = np.bincount(mask.ravel(), minlength=num_classes)
            return counts.astype(np.uint64), mask.size
    except Exception as e:
        print(f"处理 {filename} 出错: {str(e)}")
        return np.zeros(num_classes, dtype=np.uint64), 0


def compute_cls_weights_simple(mask_dir, num_classes, smooth_factor=5.0, n_jobs=-1):
    """
    简化版并行加速权重计算
    Args:
        mask_dir: mask文件夹路径
        num_classes: 类别数
        smooth_factor: 平滑因子
        n_jobs: 并行进程数（-1=自动）
    Returns:
        weights: (num_classes,) 的numpy数组
    """
    # 获取文件列表
    files = [f for f in os.listdir(mask_dir) if f.lower().endswith('.png')]
    assert len(files) > 0, "未找到PNG文件"

    # 并行处理所有文件
    results = Parallel(n_jobs=n_jobs)(delayed(process_single_file)(f, mask_dir, num_classes)
        for f in tqdm(files, desc="处理进度",file=sys.stdout))

    # 合并统计结果
    total_counts = np.zeros(num_classes, dtype=np.uint64)
    total_pixels = 0
    for counts, pixels in results:
        total_counts += counts
        total_pixels += pixels

    # 计算权重
    epsilon = 1e-7
    class_freq = (total_counts.astype(np.float64) + smooth_factor) / \
                 (total_pixels + smooth_factor * num_classes + epsilon)

    weights = 1.0 / (class_freq + epsilon)

    return weights / weights.sum()


def resize_image(image, size):
    iw, ih = image.size
    w, h = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image, nw, nh


def convert_color(image):
    """
    将图像转为RGB格式，如果是灰度类型的图像也转换为RGB格式
    :param image: 输入的图像，需要为PIL对象
    :return:
    """
    # image是一个三维的图像[w,h,rgb],并且第三维的宽度是3（r,g,b）
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def get_random_data(image, label, input_shape, jitter=.3, hue=.1, sat=.7, val=.3, random=True):
    """
    对输入图像进行数据增强处理
    :param image: features
    :param label: 标注图像
    :param input_shape: 要求输出图像的尺寸
    :param jitter: 对输入图像纵横比缩放的随机系数
    :param hue: 将图像转为HSV格式时的参数
    :param sat: 将图像转为HSV格式时的参数
    :param val: 将图像转为HSV格式时的参数
    :param random: 是否采用随机增强（对训练数据设置为True，对验证数据设置为False）
    :return:
    """
    # 转为RGB图像
    image = convert_color(image)
    label = Image.fromarray(np.array(label))

    # 获取照片尺寸
    iw, ih = image.size
    # 获取目标尺寸
    w, h = input_shape

    if not random:
        iw, ih = image.size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', [w, h], (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

        label = label.resize((nw, nh), Image.NEAREST)
        new_label = Image.new('L', [w, h], (0))
        new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
        return new_image, new_label

    else:
        # 对图像进行随机缩放
        # 生成随机纵横比
        new_ar = iw / ih * np.random.uniform(1 - jitter, 1 + jitter) / np.random.uniform(1 - jitter, 1 + jitter)
        scale = np.random.uniform(0.5, 2)
        # 计算新的高和宽
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * scale)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        # 对图像随机翻转
        # 随机生成一个数，如果小于0.5，则对图像左右反转
        flip = np.random.uniform() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # 把图像强制裁切为input_shape,并且把前面变换图像无法填充的地方使用颜色覆盖
        dx = int(np.random.uniform(0, w - nw))
        dy = int(np.random.uniform(0, h - nh))
        # 生成input_shape大小的图片，image使用灰色背景，label使用黑色背景
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), 0)
        # 把前面随机生成的图片粘贴到背景中
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        image_data = np.array(image, np.uint8)

        # 随机为图像添加高斯模糊(概率随机，添加全图)
        blur = np.random.uniform() < 0.25
        if blur:
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        # 随机旋转
        rotate = np.random.uniform() < 0.25
        if rotate:
            center = (w // 2, h // 2)
            rotation = np.random.randint(-10, 11)
            M = cv2.getRotationMatrix2D(center, -rotation, 1.0)
            image_data = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128, 128, 128))
            label = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))

        # 将图像转换到HSV色彩空间，进行随机变换，然后再转换回RGB空间
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # ---------------------------------#
        #   将图像转到HSV上
        # ---------------------------------#
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        # ---------------------------------#
        #   应用变换
        # ---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data, label


def preprocess_image(image):
    """
    对图像进行归一化操作
    :param image:
    :return:
    """
    image -= np.array([123.675, 116.28, 103.53], np.float32)
    image /= np.array([58.395, 57.12, 57.375], np.float32)
    return image


def random_seed_init(seed: int = 0):
    """
    对训练过程中的所有随机函数设置随机数种子，关闭pytorch随机优化，让训练结果可复现
    :param seed: int 随机数种子
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)

    # 设置随机数种子并采用确定性算法，让结果可以浮现，并且放弃自动调优
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.1, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.3, step_num=10):
    """
    获取学习率调度器
    :param lr_decay_type: 学习率调度器类型
    :param lr: 学习率
    :param min_lr: 最小学习率
    :param total_iters: 总迭代步数
    :param warmup_iters_ratio:
    :param warmup_lr_ratio:
    :param no_aug_iter_ratio:
    :param step_num:
    :return:
    """

    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 + math.cos(
                math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def seg_dataset_collate(batch):
    """
    定义DataLoader的组合方式
    :param batch:
    :return:
    """
    images = []
    pngs = []
    seg_labels = []

    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)

    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.array(pngs)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels


def worker_init_fn(worker_id, rank, seed):
    """
    设置DataLoader的种子
    :param worker_id:
    :param rank:
    :param seed:
    :return:
    """
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    """
    设置优化器和学习率
    :param optimizer: 优化器
    :param lr_scheduler_func: 学习率下降公式
    :param epoch:
    :return:
    """
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def ce_loss(inputs, target, cls_weights, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss


def focal_loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c).contiguous()
    temp_target = target.view(-1)

    logpt = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs,
                                                                                                 temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss


def dice_loss(inputs, target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c).contiguous(), -1)
    temp_target = target.view(n, -1, ct).contiguous()

    # --------------------------------------------#
    #   计算dice loss
    # --------------------------------------------#
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(rank, model_train, model, num_classes, cur_epoch, epoch_step, epoch_step_val, gen, gen_val,
                  total_epoch, cls_weights, cuda_enable, optimizer, fp16_enable, focal_loss_enable, dice_loss_enable,
                  scaler, eval_freq, loss_history, weight_save_freq, weight_save_path, is_save_weight,):
    total_loss = 0.0
    total_f_score = 0.0

    val_loss = 0.0
    val_f_score = 0.0

    if rank == 0:
        print(Fore.BLUE + '*' * 16 + '开始训练' + '*' * 16 + Style.RESET_ALL, flush=True)
        pbar = tqdm(total=epoch_step, desc=f'Epoch {cur_epoch + 1}/{total_epoch}', postfix=dict, mininterval=0.3,
                    position=0, leave=True, file=sys.stdout)

    # -----------------------------------训练-----------------------------------
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda_enable:
                images = images.cuda(rank)
                pngs = pngs.cuda(rank)
                labels = labels.cuda(rank)
                weights = weights.cuda(rank)

        optimizer.zero_grad()
        if not fp16_enable:
            # 前向传播
            outputs = model_train(images)

            # 计算损失
            if focal_loss_enable:
                loss = focal_loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = ce_loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss_enable:
                main_dice = dice_loss(outputs, labels)
                loss = loss + main_dice

            with torch.no_grad():
                _f_score = f_score(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

        else:
            with autocast(device_type='cuda' if cuda_enable else 'cpu'):
                outputs = model_train(images)

                # 计算损失
                if focal_loss_enable:
                    loss = focal_loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = ce_loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss_enable:
                    main_dice = dice_loss(outputs, labels)
                    loss = loss + main_dice

                with torch.no_grad():
                    _f_score = f_score(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item()
        total_f_score += _f_score.item()

        if rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if rank == 0:
        pbar.close()
        print(Fore.BLUE + '*' * 16 + '结束训练' + '*' * 16 + Style.RESET_ALL, flush=True)
        loss_history.append_loss(cur_epoch + 1, total_loss / epoch_step, total_f_score / epoch_step)

    # ---------------------------------- 评估 --------------------------------------
    if cur_epoch != 0 and cur_epoch % eval_freq == 0:
        if rank == 0:
            print(Fore.BLUE + '*' * 16 + '开始评估' + '*' * 16 + Style.RESET_ALL, flush=True)
            pbar = tqdm(total=epoch_step_val, desc=f'Epoch {cur_epoch + 1}/{total_epoch}', postfix=dict,
                        mininterval=0.3, position=0, leave=True, file=sys.stdout)

        model_train.eval()
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break

            images, pngs, labels = batch
            with torch.no_grad():
                weights = torch.from_numpy(cls_weights)
                if cuda_enable:
                    images = images.cuda(rank)
                    pngs = pngs.cuda(rank)
                    labels = labels.cuda(rank)
                    weights = weights.cuda(rank)

                outputs = model_train(images)

                if focal_loss:
                    loss = focal_loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = ce_loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice = dice_loss(outputs, labels)
                    loss = loss + main_dice

                _f_score = f_score(outputs, labels)

                val_loss += loss.item()
                val_f_score += _f_score.item()

                if rank == 0:
                    pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                        'f_score': val_f_score / (iteration + 1),
                                        'lr': get_lr(optimizer)})
                    pbar.update(1)

        if rank == 0:
            pbar.close()
            print(Fore.BLUE + '*' * 16 + '结束评估' + '*' * 16 + Style.RESET_ALL, flush=True)
            loss_history.append_val_loss(cur_epoch + 1, val_loss / epoch_step_val, val_f_score / epoch_step_val)

    if rank == 0:
        print(Fore.BLUE + f'Epoch: {cur_epoch + 1} / {total_epoch}' + Style.RESET_ALL, flush=True)
        print(f'Total Loss: {total_loss / epoch_step:.3f}, Total f_score: {total_f_score / epoch_step: .3f}',
              flush=True)
        if cur_epoch != 0 and cur_epoch % eval_freq == 0:
            print(f'Val Loss: {val_loss / epoch_step_val:.3f}, Val f_score: {val_f_score / epoch_step_val:.3f}',
                  flush=True)

            # -------------------------------保存最好的权重文件------------------------
            best_weight_save_path = os.path.join(weight_save_path, 'best_weight')
            if not os.path.exists(best_weight_save_path):
                os.makedirs(best_weight_save_path)
            if is_save_weight:
                if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
                    torch.save(model.state_dict(), os.path.join(best_weight_save_path, 'best_val_loss.pth'))

                if (val_f_score / epoch_step_val) >= max(loss_history.val_f_scores):
                    torch.save(model.state_dict(), os.path.join(best_weight_save_path, 'best_val_f_score.pth'))

        # --------------------------------------保存权重(每一周期都保存)---------------------------------
        if (cur_epoch % weight_save_freq == 0 and cur_epoch != 0) or cur_epoch + 1 == total_epoch:
            epoch_save_path = os.path.join(weight_save_path, 'every_epoch')
            if not os.path.exists(epoch_save_path):
                os.makedirs(epoch_save_path)
            torch.save(model.state_dict(),
                       os.path.join(epoch_save_path, f'ep{cur_epoch + 1:03d}-loss{total_loss / epoch_step:.3f}'
                                                     f'-f_score{total_f_score / epoch_step:.3f}.pth'))
