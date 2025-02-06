import argparse
from colorama import Fore, Style

from model.predict import predict
from model.preprocess_data import preprocess_data
from model.train import train_controller
from utils.config_reader import ConfigReader

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
    preprocess_data()

    if config.is_train():
        train_controller()

    if config.is_predict():
        predict()

if __name__ == '__main__':
    main()
