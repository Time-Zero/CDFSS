import PIL
import numpy as np
import cv2
import torch
from PIL import Image
import random

from util.tools.utils_model import EnumOptimizer
import colorsys

from colorama import Fore, Style


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

    # 如果照片的长和宽均比input_size要大，就先在其中随机裁切一份大小为input_shape的照片
    # 防止直接缩小丢失过多的细节
    if iw > w and ih > h:
        image, label = random_crop_image(image, label, input_shape)

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


def random_seed_init(seed: int = 0):
    """
    对训练过程中的所有随机函数设置随机数种子
    :param seed: int 随机数种子
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pretrained_weight_load(model_dict: dict, weight_path: str, device: torch.device):
    """
    加载预训练权重
    :param model_dict: 模型state_dict
    :param weight_path: 权重存放路径
    :param device: 将权重映射到哪个设备上(gpu or cpu)
    :return:
    """
    pretrained_dict = torch.load(weight_path, map_location=device, weights_only=True)
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    return load_key, no_load_key, temp_dict


def calculate_lf_fit(nbs: int, optimizer_type: EnumOptimizer, batch_size: int, init_lr: float, min_lr: float):
    """
    计算合适的学习率
    :param nbs: normal batch size 用于和输入的batch_size比较
    :param optimizer_type: 选择优化器种类
    :param batch_size: 批大小
    :param init_lr: 初始化学习率
    :param min_lr: 最小学习率
    :return:
    """
    lr_limit_max = 1e-4 if optimizer_type in [EnumOptimizer.ADAM, EnumOptimizer.ADAMW] else 5e-2
    lr_limit_min = 3e-5 if optimizer_type in [EnumOptimizer.ADAM, EnumOptimizer.ADAMW] else 5e-4
    init_lr_fit = min(max(batch_size / nbs * init_lr, lr_limit_min), lr_limit_max)
    min_lr_fit = min(max(batch_size / nbs * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    return init_lr_fit, min_lr_fit


def gray_image_palette_add(gray_image: PIL.Image.Image, num_classes: int, auto_colored: bool,
                           color_map: list = []) -> PIL.Image.Image:
    """
    为灰度图像添加调色板配置文件，使其可以显示为伪色彩图像
    :param gray_image: 要转为伪彩色图像的灰色图像
    :param num_classes: 总的类别
    :param auto_colored: 是否启用自动着色
    :param color_map: 手动指定的[灰度：颜色] 映射表 (启用自动着色后将失效)
    :return: 单通道伪彩色图片
    """
    # 颜色映射基表（基表用于给每一个灰度值都设置一个映射，即使灰度没有出现，如果没有基表可能会让图片从8位被优化到更低的位数）
    base_color_map = [
        [i, i, i] for i in range(256)]

    if auto_colored:
        # 自动色彩映射
        hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
        color_map = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        color_map = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), color_map))
    else:
        # 手动指定映射表
        color_map_len = len(color_map)
        assert color_map_len <= 256, "色彩映射表范围超出灰度空间"
        assert num_classes == color_map_len, "指定的[灰度:颜色]映射表长度和总类别数不一致"

    for i in range(num_classes):
        base_color_map[i] = color_map[i]

    palette = []
    for it in base_color_map:
        palette.extend(it)

    gray_image.putpalette(palette)
    return gray_image


def resize_image(image: PIL.Image, size):
    """
    predict模块中使用的图像缩放
    :param image: 待处理的图像
    :param size: 目标尺寸
    :return:
    """
    image_w, image_h = image.size
    w, h = size

    if (image_w > (w * 2) and image_h > (h * 2)) or (image_w < w /2 and image_h < h/2):
        print(Fore.YELLOW + '注意: predict中输入的图像和目标尺寸差距过大, 可能会丢失细节' + Style.RESET_ALL)

    # 计算缩放比和新的长宽
    scale = min(w / image_w, h / image_h)
    new_w = int(image_w * scale)
    new_h = int(image_h * scale)

    # 将原来的图片缩放为新的长宽
    image = image.resize((new_w, new_h), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - new_w) // 2, (h - new_h) // 2))

    return new_image, new_w, new_h


def random_crop_image(feature, label, size):
    """
    将feature和label中对应的随机位置切分大小为size的图片
    :param feature: feature照片
    :param label: label照片
    :param size: 最后要得到的照片大小
    :return:
    """
    iw, ih = feature.size
    nw, nh = size

    start_x = random.randint(0, iw - nw)
    start_y = random.randint(0, ih - nh)

    crop_feature = feature.crop((start_x, start_y, start_x + nw, start_y + nh))
    crop_label = label.crop((start_x, start_y, start_x + nw, start_y + nh))
    return crop_feature, crop_label
