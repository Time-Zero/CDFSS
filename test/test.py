import numpy as np
from PIL import Image

from utils.utils_dataset import count_unique_gray_levels
import cv2

if __name__ == '__main__':
    images_path = 'D:\毕设\数据集\CHASEDB1_VOC\SegmentationClass'
    num = count_unique_gray_levels(images_path)
    print(num)
    # image = Image.open("D:\毕设\数据集\CHASEDB1_VOC\SegmentationClass\Image_01L_1stHO.png")
    # image_np = np.array(image)
    # print(image)
