import shutil

from colorama import Fore, Style

from utils.config_reader import ConfigReader
from utils.utils_dataset import *


def preprocess_data():
    config = ConfigReader()

    dataset_path = config.get_dataset_path()
    if not os.path.exists(dataset_path):
        raise AssertionError('数据集地址错误，无法找到数据集')

    # -------------------------是否重新划分数据集----------------------------
    need_divide = config.need_divide()
    if need_divide:
        print(Fore.GREEN + '启用数据重划分，将执行该操作' + Style.RESET_ALL)
        divide_percent = config.get_divide_percent()
        dataset_divide(dataset_path, divide_percent)
        print(Fore.GREEN + '数据重划分完成' + Style.RESET_ALL)

    # -----------------------是否需要裁剪图片-----------------------------
    need_crop = config.need_crop()
    if need_crop:
        print(Fore.GREEN + '启用图片裁剪，将执行该操作' + Style.RESET_ALL)
        imagesets_path = os.path.join(dataset_path, 'ImageSets')
        jpegimages_path = os.path.join(dataset_path, 'JPEGImages')
        segmentations_path = os.path.join(dataset_path, 'SegmentationClass')
        if not os.path.exists(imagesets_path) or not os.path.exists(jpegimages_path) or not os.path.exists(
                segmentations_path):
            raise FileNotFoundError('数据集结果不符合Voc数据集格式')

        # 创建与处理过的数据集保存路径
        save_path = os.path.join("./results", 'processed_dataset')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            shutil.rmtree(save_path)
            os.makedirs(save_path)
        config.set_dataset_path(save_path)  # 只要经过处理，那么dataset_path就会改变

        shutil.copytree(imagesets_path, os.path.join(save_path, 'ImageSets'))

        jpeg_save_path = os.path.join(save_path, 'JPEGImages')
        segmentation_save_path = os.path.join(save_path, 'SegmentationClass')
        if not os.path.exists(jpeg_save_path):
            os.makedirs(jpeg_save_path)
        if not os.path.exists(segmentation_save_path):
            os.makedirs(segmentation_save_path)

        crop_size = config.get_crop_size()
        with open(os.path.join(imagesets_path, 'Segmentation\\trainval.txt'), 'r', encoding='utf-8') as f:
            file_list = f.readlines()
        file_list = [line.strip() for line in file_list]
        pbar = tqdm(total=len(file_list), position=0, leave=True, unit='images')
        for file_name in file_list:
            pbar.update(1)

            try:
                feature = Image.open(os.path.join(jpegimages_path, file_name + '.jpg'))
                label = Image.open(os.path.join(segmentations_path, file_name + '.png'))
            except FileNotFoundError as e:
                continue

            result = image_crop(feature, label, crop_size)
            if result is None:
                continue
            else:
                feature, label = result

            feature.save(os.path.join(jpeg_save_path, file_name + '.jpg'))
            label.save(os.path.join(segmentation_save_path, file_name + '.png'))

        pbar.close()
        print(Fore.GREEN + '图片裁剪完成' + Style.RESET_ALL)

    # ------------------------ 是否需要将分割图片中的灰度图片进行着色操作来提高可读性--------------------------------
    need_color = config.preprocess_color()
    if need_color:
        print(Fore.GREEN + '启用灰度图片着色，将执行该操作' + Style.RESET_ALL)

        auto_color = config.get_auto_color()
        if auto_color:
            color_map = None
        else:
            color_map = config.get_color_map()

        gray_image_path = os.path.join(config.get_dataset_path(), 'SegmentationClass')
        gray_file_list = os.listdir(gray_image_path)
        pbar = tqdm(total=len(gray_file_list), position=0, leave=True, unit='images')

        num_classes = config.get_num_classes()
        for file_name in gray_file_list:
            pbar.update(1)

            try:
                image = Image.open(os.path.join(gray_image_path, file_name))
                if image.mode != 'L':
                    image = image.convert('L')
            except Exception as e:
                continue

            image = grayscale2colored(image, num_classes, color_map)
            image.save(os.path.join(gray_image_path, file_name))

        pbar.close()
        print(Fore.GREEN + '灰度图片着色完成' + Style.RESET_ALL)