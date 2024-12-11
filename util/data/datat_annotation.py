import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm


class DataAnnotation:
    """
    数据标注，生成类似Voc数据集中的train.txt,val.txt,trainval.txt结构
    需要数据集的feature全部存放在JPEGImages文件夹中，并且为.jpg格式
    需要数据集的label全部存放在SegmentationClass文件夹中，并且全部为.png格式
    格式为.png的label需要为单通道灰度图片或者单通道伪彩色图片
    """
    def __init__(self, data_path: str, train_rate: float, val_rate: float, test_rate: float):
        self.annotation_path = None
        self.seg_path = None
        self.jpeg_path = None
        self.data_path = data_path
        self.train_rate = train_rate
        self.val_rate = val_rate
        self.test_rate = test_rate

        if train_rate + val_rate + test_rate != 1:
            raise ValueError('数据集分配错误，请检查数据集分配比例')

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(self.data_path)
        if not os.path.isdir(self.data_path):
            raise NotADirectoryError(self.data_path)

    def do_annotation(self):
        """
        执行数据划分
        :return:
        """
        self.jpeg_path = os.path.join(self.data_path, 'JPEGImages')
        self.seg_path = os.path.join(self.data_path, 'SegmentationClass')
        self.annotation_path = os.path.join(self.data_path, 'ImageSets/Segmentation')

        if not os.path.exists(self.jpeg_path):
            raise FileNotFoundError(self.jpeg_path + '路径不存在，请检查数据集格式')
        if not os.path.exists(self.seg_path):
            raise FileNotFoundError(self.seg_path + '路径不存在，请检查数据集格式')

        # 如果文件夹不存在，就创建文件夹
        if not os.path.exists(self.annotation_path):
            os.makedirs(self.annotation_path)

        # 获取label文件夹中的所有文件
        label_files = os.listdir(self.seg_path)
        assert len(label_files) > 0, '待标注文件为空，请检查'

        pbar = tqdm(total=len(label_files), desc='正在遍历标注文件', postfix='遍历进度')
        temp_files = []
        # 遍历文件
        for label_file in label_files:
            # 获取文件类型
            _, ext = os.path.splitext(label_file)
            # 如果是png文件
            if ext == '.png':
                # 检查标注和特征图是否都存在
                feature_file_path = os.path.join(self.jpeg_path, label_file.replace('.png', '.jpg'))
                label_file_path = os.path.join(self.seg_path, label_file)
                if os.path.isfile(feature_file_path):
                    with Image.open(label_file_path) as img:
                        # 如果文件是8位图 （灰度图像或者是伪色彩图像）
                        img_array = np.array(img)
                        if img_array.dtype == np.uint8 and len(img_array.shape) == 2:
                            file_name, _ = os.path.splitext(os.path.basename(label_file))
                            temp_files.append(os.path.basename(file_name))

            pbar.update(1)

        # 随机打乱数据
        random.shuffle(temp_files)

        # 计算三个数据集的长度
        temp_files_len = len(temp_files)
        train_data_len = int(self.train_rate * temp_files_len)
        val_data_len = int(self.val_rate * temp_files_len)

        # 划分数据集
        print('---------------------划分数据集中-----------------------')
        train_files = temp_files[:train_data_len]
        val_files = temp_files[train_data_len:train_data_len + val_data_len]
        test_files = temp_files[train_data_len + val_data_len:]

        train_files.sort()
        val_files.sort()
        test_files.sort()

        print('---------------------写入划分文件-----------------------')
        train_files_path = os.path.join(self.annotation_path, 'train.txt')
        val_files_path = os.path.join(self.annotation_path, 'val.txt')
        test_files_path = os.path.join(self.annotation_path, 'test.txt')
        train_val_files_path = os.path.join(self.annotation_path, 'trainval.txt')

        with open(train_files_path, 'w') as f:
            for train_file in train_files:
                f.write(train_file + '\n')

        with open(val_files_path, 'w') as f:
            for val_file in val_files:
                f.write(val_file + '\n')

        with open(test_files_path, 'w') as f:
            for test_file in test_files:
                f.write(test_file + '\n')

        with open(train_val_files_path, 'w') as f:
            for temp_file in temp_files:
                f.write(temp_file + '\n')

        print('结束划分')
        return train_files, val_files, test_files


if __name__ == '__main__':
    floodnet_path = 'E:\\毕设\\Cross_Domain_Few_Shot_Segmentation_System\\data\\train_data\\FloodNet_Voc_Format'
    data_annotation = DataAnnotation(floodnet_path, 0.7, 0.15, 0.15)
    train_files, val_files, test_files = data_annotation.do_annotation()
