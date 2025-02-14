import argparse
import os.path

from colorama import Fore, Style

from model.predict import predict
from model.preprocess_data import preprocess_data
from model.train import train_controller
from utils.config_reader import ConfigReader
from utils.utils_dataset import count_unique_gray_levels


def main():
    parser = argparse.ArgumentParser(description='CDFSS: 一个跨域小样本模型训练系统', add_help=True, epilog='请指定参数运行')

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

    if config.is_auto_get_num_name():
        print(Fore.BLUE + 16 * '*' + '启用自动灰度范围推理' + 16 * '*' + Style.RESET_ALL)
        dataset_path = config.get_dataset_path()
        segmentation_path = os.path.join(dataset_path, 'SegmentationClass')
        num_gray_levels, gray_levels_set = count_unique_gray_levels(segmentation_path)
        print(Fore.BLUE + 16 * '*' + '自动灰度范围推理完成' + 16 * '*' + Style.RESET_ALL)

        # 对灰度值255进行处理
        if 255 in gray_levels_set:
            flag = input(Fore.YELLOW + '出现灰度值255，是否舍弃(Y/n): ' + Style.RESET_ALL)
            while flag != 'Y' or flag != 'n':
                flag = input(Fore.RED + '输入错误，请重新输入: ' + Style.RESET_ALL)

            if flag == 'Y':
                gray_levels_set.remove(255)

        # 根据灰度值自动生成name_classes
        gray_levels_list = sorted(list(gray_levels_set))
        name_classes = [f'{i}-{value}' for i, value in enumerate(gray_levels_list)]
        config.set_name_classes(name_classes)
        config.set_num_classes(num_gray_levels)

    if config.is_preprocess():
        preprocess_data()

    if config.is_train():
        train_controller()

    if config.is_predict():
        predict()

if __name__ == '__main__':
    main()
