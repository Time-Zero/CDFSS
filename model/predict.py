import os
import sys

from PIL import Image
from colorama import Fore, Style
from tqdm import tqdm

from model.segformer_predict import SegformerPredict
from utils.config_reader import ConfigReader
from utils.utils_dataset import grayscale2colored
from utils.utils_predict import compute_miou, show_results


def predict():
    config = ConfigReader()

    need_color = config.need_color()
    colors_map = None
    if need_color:
        if config.get_auto_color():
            colors_map = None
        else:
            colors_map = config.get_color_map()

    save_path = config.get_pred_res_save_path()
    miou_save_path = os.path.join(save_path, 'miou_out')
    pred_img_save_path = os.path.join(save_path, 'pred_img_out')
    if not os.path.exists(miou_save_path):
        os.makedirs(miou_save_path)
    if not os.path.exists(pred_img_save_path):
        os.makedirs(pred_img_save_path)

    dataset_path = config.get_dataset_path()
    label_path = os.path.join(dataset_path, 'SegmentationClass')
    image_ids = open(os.path.join(dataset_path, "ImageSets\\Segmentation\\test.txt")).read().splitlines()

    print(Fore.BLUE + '*' * 16 + '加载预测模型中' + '*' * 16 + Style.RESET_ALL)
    model = SegformerPredict(model_path=config.get_model_path(), num_class=config.get_num_classes(),
                             backbone=config.get_phi(),
                             input_shape=config.get_input_size(), cuda=config.pred_cuda_enable())
    print(Fore.BLUE + '*' * 16 + '模型加载成功' + '*' * 16 + Style.RESET_ALL)

    print(Fore.BLUE + '*' * 16 + '获取预测图片中' + '*' * 16 + Style.RESET_ALL)
    num_classes = config.get_num_classes()
    for image_id in tqdm(image_ids, position=0, leave=True, file=sys.stdout):
        image_path = os.path.join(dataset_path, 'JPEGImages', image_id + '.jpg')
        image = Image.open(image_path)
        image = model.get_miou_png(image)
        if need_color:
            image = grayscale2colored(image, num_classes, colors_map)
        image.save(os.path.join(pred_img_save_path, image_id + '.png'))
    print(Fore.BLUE + '*' * 16 + '获取预测图片完成' + '*' * 16 + Style.RESET_ALL)

    hist, IoUs, PA_Recall, Precision = compute_miou(label_path, pred_img_save_path, image_ids, num_classes)
    show_results(miou_save_path, hist, IoUs, PA_Recall, Precision, config.get_name_classes())
