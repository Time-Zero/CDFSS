import math
import os
from enum import Enum
from functools import partial

import torch
from torch import optim, nn
from torch.cuda.amp import autocast
from tqdm import tqdm

from util.tools.loss import focal_loss, ce_loss, dice_loss
from util.tools.utils_metrics import f_score


class EnumOptimizer(Enum):
    ADAM = 1
    ADAMW = 2
    SGD = 3


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


def get_lr(optimizer):
    """
    获取学习率
    :param optimizer:
    :return:
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def optimizer_select(optimizer_type: EnumOptimizer, model: nn.Module, init_lr_fit: float, momentum: float,
                     weight_decay: float) -> torch.optim.Optimizer:
    """
    选择和初始化优化器
    :param optimizer_type: 优化器种类
    :param model: 模型
    :param init_lr_fit: 初始化学习率
    :param momentum: 一阶动量的衰减率
    :param weight_decay: 权重衰减参数
    :return:
    """
    optimizer = {
        EnumOptimizer.ADAM: optim.Adam(model.parameters(), init_lr_fit, betas=(momentum, 0.999),
                                       weight_decay=weight_decay),
        EnumOptimizer.ADAMW: optim.AdamW(model.parameters(), init_lr_fit, betas=(momentum, 0.999),
                                         weight_decay=weight_decay),
        EnumOptimizer.SGD: optim.SGD(model.parameters(), init_lr_fit, momentum=momentum, nesterov=True,
                                     weight_decay=weight_decay)
    }[optimizer_type]

    return optimizer


def fit_one_epoch(model_train, model, optimizer, num_classes, cur_epoch, epoch_step, epoch_step_val, gen, gen_val,
                  total_epoch,
                  cuda_enable, focal_loss_flag, dice_loss_flag, cls_weights, fp16, scaler):
    total_loss = 0.0
    total_f_score = 0.0

    val_loss = 0.
    val_f_score = 0.

    print('开始训练')
    pbar = tqdm(total=epoch_step, desc=f'Epoch {cur_epoch + 1}/{total_epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda_enable:
                images = images.cuda()
                pngs = pngs.cuda()
                labels = labels.cuda()
                weights = weights.cuda()

        optimizer.zero_grad()

        if not fp16:
            # 如果没有使用混合精度
            # 前向传播
            outputs = model_train(images)

            # 计算损失
            if focal_loss_flag:
                loss = focal_loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = ce_loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss_flag:
                main_dice = dice_loss(outputs, pngs)
                loss = loss + main_dice

            with torch.no_grad():
                # 计算f_score
                _f_score = f_score(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

        else:
            # 启用混合精度
            with autocast():
                outputs = model_train(images)

                if focal_loss_flag:
                    loss = focal_loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = ce_loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss_flag:
                    main_dice = dice_loss(outputs, pngs)
                    loss = loss + main_dice

                with torch.no_grad():
                    # 计算f_score
                    _f_score = f_score(outputs, labels)

            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        total_f_score += _f_score.item()

        pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                            'f_score': total_f_score / (iteration + 1),
                            'lr': get_lr(optimizer)})
        pbar.update(1)

    pbar.close()
    print('训练结束')

    print('开始评估')
    pbar = tqdm(total=epoch_step_val, desc=f'Epoch {cur_epoch + 1}/{total_epoch}', postfix=dict, mininterval=0.3)
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break

        images, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda_enable:
                images = images.cuda()
                pngs = pngs.cuda()
                labels = labels.cuda()
                weights = weights.cuda()

            # 前向传播
            outputs = model_train(images)

            # 计算损失
            if focal_loss_flag:
                loss = focal_loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = ce_loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss_flag:
                main_dice = dice_loss(outputs, pngs)
                loss = loss + main_dice

            _f_score = f_score(outputs, labels)

            val_loss += loss.item()
            val_f_score += _f_score.item()

        pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                            'f_score': val_f_score / (iteration + 1),
                            'lr': get_lr(optimizer)})
        pbar.update(1)

    pbar.close()
    print('结束评估')
    print('Epoch:' + str(cur_epoch + 1) + '/' + str(total_epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
    torch.save(model.state_dict(), os.path.join('.', "last_epoch_weights.pth"))
