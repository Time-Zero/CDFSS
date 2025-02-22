import colorsys
import os
import random
import sys
from copy import deepcopy

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def dataset_divide(dataset_path: str, divide_per: list) -> None:
    """
    数据集切分
    :param dataset_path: 数据集路径
    :param divide_per: 训练集比例 [train, val, test]
    :return:
    """
    train_per, val_per, test_per = divide_per
    train_per = round(train_per, 2)
    val_per = round(val_per, 2)
    test_per = round(test_per, 2)

    if not (abs(train_per + val_per + test_per - 1) < 1e-9):
        raise ValueError('train_per + val_per + test_per 不为 1.!')

    imageset_path = os.path.join(dataset_path, 'ImageSets/Segmentation')
    feature_path = os.path.join(dataset_path, 'JPEGImages')
    label_path = os.path.join(dataset_path, 'SegmentationClass')

    # 检查文件路径
    if not os.path.exists(imageset_path):
        os.makedirs(imageset_path)

    if not os.path.exists(feature_path) or not os.path.exists(label_path):
        raise FileNotFoundError('数据集不是Voc结构！！')

    # 读取文件列表
    label_file_list = os.listdir(label_path)
    pbar = tqdm(total=len(label_file_list), position=0, leave=True, unit='images', file=sys.stdout)

    res = []
    for label_file_name in label_file_list:
        # 如果不是文件
        pbar.update(1)
        if not os.path.isfile(os.path.join(label_path, label_file_name)):
            continue

        file_base_name = os.path.splitext(label_file_name)[0]
        # 如果png文件没有对应的jpg文件
        if not os.path.exists(os.path.join(feature_path, file_base_name + '.jpg')):
            continue

        res.append(file_base_name)
    pbar.close()

    res_len = len(res)
    if res_len == 0:
        raise ValueError('无法划分，请检查数据集结构，确保JPEGImages和SegmentationClass中文件名对应')
    train_list_len = int(res_len * train_per)
    val_list_len = int(res_len * val_per)

    trainval_list = deepcopy(res)
    random.shuffle(res)
    train_list = res[: train_list_len]
    val_list = res[train_list_len: train_list_len + val_list_len]
    test_list = res[train_list_len + val_list_len:]

    with open(os.path.join(imageset_path, 'trainval.txt'), 'w') as f:
        trainval_list.sort()
        for item in trainval_list:
            f.write(item + '\n')

    with open(os.path.join(imageset_path, 'train.txt'), 'w') as f:
        train_list.sort()
        for item in train_list:
            f.write(item + '\n')

    with open(os.path.join(imageset_path, 'val.txt'), 'w') as f:
        val_list.sort()
        for item in val_list:
            f.write(item + '\n')

    with open(os.path.join(imageset_path, 'test.txt'), 'w') as f:
        test_list.sort()
        for item in test_list:
            f.write(item + '\n')


def image_crop(feature: Image, label: Image, target_size: list):
    """
    裁剪图片
    :param feature: feature照片(为.jpg格式)
    :param label: label照片(为.png格式)
    :param target_size: 想要切分到的尺寸
    :return: 返回PIL.Image格式的feature, label；如果输入图片尺寸比target_size小，将返回None
    """
    iw, ih = feature.size
    nw, nh = target_size

    if (iw < nw or ih < nh) or (feature.size != label.size):
        return None

    start_x = random.randint(0, iw - nw)
    start_y = random.randint(0, ih - nh)

    crop_feature = feature.crop((start_x, start_y, start_x + nw, start_y + nh))
    crop_label = label.crop((start_x, start_y, start_x + nw, start_y + nh))
    return crop_feature, crop_label


def grayscale2colored(image: Image, num_classes: int, color_map: list) -> Image:
    """
    将灰度图像转伪彩色图像
    :param image: 待转换的灰度图像
    :param num_classes: 总类别
    :param color_map: [灰度:彩色] 映射关系, 如果输入None将启用自动色彩映射
    :return: 伪彩色图像
    """

    if color_map is None:
        hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
        color_map = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        color_map = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), color_map))
    else:
        assert (len(color_map) == num_classes), 'color_map（色彩映射关系）和类别数量不匹配'
        assert (len(color_map) <= 256), 'color_map（色彩映射关系）超过灰度范围'

        base_color_map = [
            [i, i, i] for i in range(256)]
        for i in range(num_classes):
            base_color_map[i] = color_map[i]

        color_map = base_color_map

    palette = []
    for it in color_map:
        palette.extend(it)

    image.putpalette(palette)
    return image

def count_unique_gray_levels(images_path):
    """
    统计灰度值数量
    :param images_path: 存放图像文件夹
    :return: 灰度值数量,灰度值集合
    """
    unique_gray_levels = set()

    for filename in tqdm(os.listdir(images_path), desc="Get Gray Levels", position=0, leave=True, file=sys.stdout):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(images_path, filename)
            image = Image.open(image_path)

            if image is not None:
                image_np = np.array(image, dtype=np.uint8)
                unique_levels = np.unique(image_np)
                unique_gray_levels.update(unique_levels)

    return len(unique_gray_levels), unique_gray_levels

def bin_image_convert(image_path):
    flag = False

    for filename in tqdm(os.listdir(image_path), desc="Get Mode", position=0, leave=True, file=sys.stdout):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(image_path, filename)
            image = Image.open(file_path)

            if image is not None:
                if image.mode == '1':
                    image = image.convert('L')
                    image_array = np.array(image)

                    image_array[image_array == 255] = 1
                    image = Image.fromarray(image_array, mode='L')

                    image.save(file_path)

                    if not flag:
                        flag = True

            image.close()
    return flag