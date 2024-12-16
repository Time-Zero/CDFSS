import os

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

from util.tools.utils import get_random_data, preprocess_image


class SegmentationDataset(Dataset):
    """
    自定义的DataLoader
    """
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        """

        :param annotation_lines: 标注数据行
        :param input_shape:
        :param num_classes: 类别数量
        :param train: 是否使用训练模式，如果使用训练模式则输出随机数据
        :param dataset_path: 数据集位置
        """
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.length = len(self.annotation_lines)
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path
        self.feature_path = None
        self.label_path = None

        self.feature_path = os.path.join(self.dataset_path, "JPEGImages")
        self.label_path = os.path.join(self.dataset_path, "SegmentationClass")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]

        # 从文件中获取图像
        jpg = Image.open(os.path.join(self.feature_path, name + '.jpg'))
        png = Image.open(os.path.join(self.label_path, name + '.png'))

        # 数据增强
        jpg, png = get_random_data(image=jpg, label=png, input_shape=self.input_shape, random=self.train)

        jpg = np.transpose(preprocess_image(np.array(jpg, np.float64)), [2, 0, 1])
        png = np.array(png)
        png[png >= self.num_classes] = self.num_classes

        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, png, seg_labels

