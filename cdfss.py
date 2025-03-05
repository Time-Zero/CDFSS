import argparse
import os.path

import torch.cuda
from colorama import Fore, Style

from model.predict import predict
from model.preprocess_data import preprocess_data
from model.train import train_controller
from utils.config_reader import ConfigReader
from utils.utils_common import func_print
from utils.utils_dataset import count_unique_gray_levels, bin_image_convert


def main():
    parser = argparse.ArgumentParser(description='CDFSS: 一个跨域小样本模型训练系统', add_help=True,
                                     epilog='请指定参数运行')
    parser.add_argument('-d', '--default', action='store_true', help='默认模式,手动指定所有配置')
    parser.add_argument('-c', '--config', type=str, help='指定配置文件')
    args = parser.parse_args()

    if args.default:
        print(Fore.BLUE + '默认配置文件运行' + Style.RESET_ALL)
    elif args.config:
        print(Fore.BLUE + f'指定配置文件: {args.config}' + Style.RESET_ALL)
        ConfigReader().set_conf_path(args.config)
    else:
        print(Fore.RED + '请指定运行参数，或输入-h/--help获取帮助' + Style.RESET_ALL)
        exit(1)

    config = ConfigReader()

    # 判断是不是二值图
    func_print('blue', 16, '识别SegmentationImage类别中')
    is_bin_image = bin_image_convert(os.path.join(config.get_dataset_path(), 'SegmentationClass'))
    gray_levels_set = None
    num_gray_levels = None
    if is_bin_image:
        func_print('yellow', 0, '检测到二值图，将转为灰度图')
        gray_levels_set = {0, 1}
        num_gray_levels = 2
    func_print('blue', 16, '识别完成')

    # 如果不是二值图并且启用了自动灰度推理
    if config.is_auto_get_num_name():
        if not is_bin_image:
            func_print('blue', 16, '启用自动灰度范围推理')
            dataset_path = config.get_dataset_path()
            segmentation_path = os.path.join(dataset_path, 'SegmentationClass')
            num_gray_levels, gray_levels_set = count_unique_gray_levels(segmentation_path)

            # 对灰度值255进行处理
            if 255 in gray_levels_set:
                gray_levels_set.remove(255)
                num_gray_levels -= 1

            print(f'Segmentation图像中，包括的灰度值为: {gray_levels_set}')
            func_print('blue', 16, '灰度自动推理完成')

        # 根据灰度值自动生成name_classes
        gray_levels_list = sorted(list(gray_levels_set))
        name_classes = [f'{i}-{value}' for i, value in enumerate(gray_levels_list)]
        config.set_name_classes(name_classes)
        config.set_num_classes(num_gray_levels)

    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        func_print('yellow', 16, f'检测到的可用gpu数量: {gpu_count}，已自动关闭cuda')
        config.set_cuda_enable(False)
        config.set_pred_cuda_enable(False)

    if config.is_preprocess():
        preprocess_data()

    if config.is_train():
        train_controller()

    if config.is_predict():
        predict()


if __name__ == '__main__':
    main()
