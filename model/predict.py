from PIL import Image

from model.segformer.segformer_predict import SegformerPredict
from util.config.config_reader import *


def predict():
    config = ConfigReader()
    mode = config.get_predict_mode()
    cuda_enable = config.get_predict_cuda_enable()
    model_weight_path = config.get_predict_weight_path()
    num_classes = config.get_predict_num_classes()
    backbone = config.get_predict_backbone()
    input_shape = config.get_predict_input_shape()
    mix_type = config.get_predict_mix_type()
    auto_colored = config.auto_colored()
    count = config.is_count_pixel()

    if count:
        classes_name = config.get_classes_name()
    else:
        classes_name = None

    if auto_colored:
        color_map = None
    else:
        color_map = config.get_color_map()

    if mode == 'predict':
        segformer = SegformerPredict(cuda=cuda_enable, model_path=model_weight_path, num_classes=num_classes,
                                     auto_colored=auto_colored, color_map=color_map, backbone=backbone,
                                     input_shape=input_shape, mix_type=mix_type)

        while True:
            img_path = input('输入图像文件路径(输入^来退出): ')
            if img_path == '^':
                break

            try:
                img = Image.open(img_path)
            except Exception as e:
                print(f"无法打开文件: {e}")
                continue
            else:
                r_image = segformer.detect_image(img, count, classes_name)
                r_image.show()

if __name__ == '__main__':
    predict()

