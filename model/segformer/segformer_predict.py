import colorsys
import copy

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model.segformer.segformer import SegFormer
from util.tools.utils import convert_color, resize_image, preprocess_image


class SegformerPredict(object):
    def __init__(self, cuda, model_path, num_classes, auto_colored, color_map, backbone, input_shape, mix_type):
        self.cuda = cuda
        self.model_path = model_path
        self.num_classes = num_classes
        self.auto_colored = auto_colored
        self.color_map = color_map
        self.backbone = backbone
        self.input_shape = input_shape
        self.mix_type = mix_type

        self.net = None

        # ------------------是否启用自动色彩映射----------------
        if self.auto_colored:
            # 通过num_classes计算一个自动的色彩映射
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            color_map = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.color_map = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), color_map))
        else:
            # 使用指定的色彩映射
            color_map_len = len(self.color_map)
            assert self.num_classes == color_map_len, '色彩映射表和总类别长度不一致'
            assert color_map_len <= 256, '色彩映射表长度超过灰度范围'

        self.generate()

    def generate(self):
        """
        按照参数初始化模型
        :return:
        """
        device = torch.device('cuda' if torch.cuda.is_available() and self.cuda else 'cpu')
        self.net = SegFormer(num_classes=self.num_classes, phi=self.backbone, pretrained=False)
        self.net.load_state_dict(torch.load(self.model_path, map_location=device, weights_only=True))
        self.net = self.net.eval()
        self.net = torch.nn.DataParallel(self.net)
        self.net = self.net.cuda()

    def detect_image(self, image, count=False, name_classes=None):
        # 转换为rgb图片
        image = convert_color(image)

        old_img = copy.deepcopy(image)
        original_h = np.array(image).shape[0]
        original_w = np.array(image).shape[1]

        # 将图片缩放并且添加灰条
        image_data, nw, nh = resize_image(image, (self.input_shape[0], self.input_shape[1]))
        # 加上batch_size的维度
        image_data = np.expand_dims(np.transpose(preprocess_image(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            pr = self.net(images)[0]
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            # --------------------------------------#
            #   将灰条部分截取掉
            # --------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            # ---------------------------------------------------#
            #   进行图片的resize
            # ---------------------------------------------------#
            pr = cv2.resize(pr, (original_h, original_w), interpolation=cv2.INTER_LINEAR)
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = pr.argmax(axis=-1)

        if count:
            # 计算不同类别的像素点的数量
            classes_nums = np.zeros(self.num_classes)
            total_points_num = original_h * original_w

            print('-' * 63)
            print("|%25s | %15s | %15s|" % ("Key", "Value", "Ratio"))
            print('-' * 63)

            for i in range(self.num_classes):
                num = np.sum(pr == i)
                ratio = num / total_points_num
                if num > 0:
                    if num > 0:
                        print("|%25s | %15s | %14.2f%%|" % (str(name_classes[i]), str(num), ratio))
                        print('-' * 63)
                    classes_nums[i] = num
            print("classes_nums:", classes_nums)

        if self.mix_type == 0:
            seg_img = np.reshape(np.array(self.color_map, np.uint8)[np.reshape(pr, [-1])],
                                 [original_h, original_w, -1])
            # 将新图片转为Image
            image = Image.fromarray(seg_img)
            # 将新图片和原图进行混合
            image = Image.blend(old_img, image, 0.7)
        elif self.mix_type == 1:
            seg_img = np.reshape(np.array(self.color_map, np.uint8)[np.reshape(pr, [-1])],
                                 [original_h, original_w, -1])
            image = Image.fromarray(np.uint8(seg_img))
        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            image = Image.fromarray(np.uint8(seg_img))

        return image