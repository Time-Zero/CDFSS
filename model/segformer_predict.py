import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from model.segformer import SegFormer
from utils.utils_model import convert_color, resize_image, preprocess_image


class SegformerPredict:
    def __init__(self, model_path, num_class, backbone, input_shape, cuda):
        self.net = None
        self.model_path = model_path
        self.num_class = num_class
        self.backbone = backbone
        self.input_shape = input_shape
        self.cuda = cuda

        self.generate_model()

    def generate_model(self):
        self.net = SegFormer(num_classes=self.num_class, phi=self.backbone, pretrained=False, backbone_weight_path='')
        device = torch.device('cuda' if self.cuda else 'cpu')

        self.net.load_state_dict(torch.load(self.model_path, map_location=device, weights_only=True))
        self.net = self.net.eval()

        if self.cuda:
            self.net = self.net.cuda()

    def get_miou_png(self, image):
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = convert_color(image)
        original_h = np.array(image).shape[0]
        original_w = np.array(image).shape[1]
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data, nw, nh = resize_image(image, self.input_shape)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
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
            pr = pr[int((self.input_shape[1] - nh) // 2): int((self.input_shape[1] - nh) // 2 + nh), \
                 int((self.input_shape[0] - nw) // 2): int((self.input_shape[0] - nw) // 2 + nw)]
            # ---------------------------------------------------#
            #   进行图片的resize
            # ---------------------------------------------------#
            pr = cv2.resize(pr, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image
