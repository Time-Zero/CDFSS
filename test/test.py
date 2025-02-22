from pickletools import uint8

import numpy as np
from PIL import Image

from utils.utils_dataset import count_unique_gray_levels, grayscale2colored
import cv2

if __name__ == '__main__':
    # images_path = 'D:\毕设\数据集\CHASEDB1_VOC\SegmentationClass'
    # num = count_unique_gray_levels(images_path)
    # print(num)
    path = "E:\毕设\CDFSS\data\dataset\CHASEDB1_VOC\SegmentationClass\Image_01L.png"
    image = Image.open(path)
    image = grayscale2colored(image, 2, None)
    image.save(path, mode='L')
