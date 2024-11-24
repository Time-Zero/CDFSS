import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim, nn
import torchvision
from PIL import Image
from tqdm import tqdm

# 标签中每个RGB颜色的值
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
# 标签其标注的类别
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']



def read_images(root="E:\\毕设\\训练数据\\voc2012\\VOCdevkit\\VOC2012", is_train=True, max_num = None):
    """
    读取训练图像，并且以RGB的格式存储
    :param root: 根路径
    :param is_train: 是否是训练模式（也就是是读取训练数据还是验证数据）
    :param max_num: 最大数量
    :return:
    """
    txt_file_name = f'{root}/ImageSets/Segmentation/{"train.txt" if is_train else "val.txt"}'
    with open(txt_file_name, 'r') as f:
        images = f.read().split()           # 将文件名以list形式读取

    # 如果指定了最大读取大小
    if max_num is not None:
        images = images[:min(max_num, len(images))]

    features, labels = [None] * len(images), [None] * len(images)
    for index, file_name in tqdm(enumerate(images), total=len(images)):
        # 读入数据，并且全部转为RGB形式的PIL的image
        features[index] = Image.open(f'{root}/JPEGImages/{file_name}.jpg').convert('RGB')
        labels[index] = Image.open(f'{root}/SegmentationClass/{file_name}.png').convert('RGB')

    return features, labels

def show_images(images, num_rows, num_cols, scale = 2):
    """
    将图片以指定的网格形式显示
    :param images: 图片数组
    :param num_rows: 行网格数
    :param num_cols: 列网格数
    :param scale: 缩放比例
    :return:
    """
    figure_size = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figure_size)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(images[i * num_cols + j])         # 计算这个网格应该存放的图片的索引
            axes[i][j].axes.get_xaxis().set_visible(False)      # 设置x坐标轴不可见
            axes[i][j].axes.get_yaxis().set_visible(False)      # 设置y坐标轴不可见
    plt.show()
    return axes

def label_indices(colormap, colormap2label):
    """
    构造标签矩阵
    :param colormap: 要标注label的图片
    :param colormap2label: 颜色和label映射表（一维表）
    :return: 二维图像的指定颜色打上标签
    """
    # 通过下面的步骤，将图片转换为一个[row_pixels][col_pixels][rgb_color]的np矩阵
    colormap = np.array(colormap.convert('RGB')).astype('int32')
    # 再利用这个矩阵将图片的像素从rgb三维空间映射到256 ** 3的线性空间中
    idx = ((colormap[:,:,0] * 256 + colormap[:,:,1]) * 256 + colormap[:,:,2])
    # 使用colormap2label中不同颜色对应的标签，向图片对应的颜色标注标签
    return colormap2label[idx]

def rand_crop(feature, label, height, width):
    """
    为了使得图像符合模型的输入，我们对图像进行随机裁剪
    :param feature: 要裁剪的原图像
    :param label: 图像标注
    :param height: 图像高
    :param width: 图像宽
    :return: 裁剪后的子图和子标注
    """
    # 生成一个在原图像上随机裁剪出一个子图像的位置参数； i、j：起始像素，h、w：在起始像素后多长距离
    i, j, h, w = torchvision.transforms.RandomCrop.get_params(feature, output_size=(height, width))
    # 生成子图
    feature = torchvision.transforms.functional.crop(feature, i, j, h, w)
    label = torchvision.transforms.functional.crop(label, i, j, h, w)
    return feature, label

class VOCSegDataset(torch.utils.data.Dataset):
    """
    自定义的数据加载器，实现了对不符合要求的图像的移除和作为feature图像的色彩归一化
    """
    def __init__(self, is_train, crop_size, voc_dir, colormap2label, max_num = None):
        """

        :param is_train:
        :param crop_size: (h,w)
        :param voc_dir:
        :param clormap2label:
        :param max_num:
        """
        # 图像颜色的平均值和标准差(是先将图像转换为张量后再计算)，这是从大规模图像数据中计算的来的，理论上可以使用在大部分图像处理中
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.tsf = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)])
        self.crop_size = crop_size
        feature, label = read_images(voc_dir, is_train=is_train, max_num=max_num)
        # 通过自定义的过滤器，移除尺寸小于要求的图像
        self.feature = self.filter(feature)
        self.label = self.filter(label)
        self.colormap2label = colormap2label
        print(f'read {len(self.feature)} valid images')

    def filter(self, images):
        return [image for image in images if (image.size[1] >= self.crop_size[0] and image.size[0] >= self.crop_size[1])]

    def __getitem__(self, index):
        feature, label = rand_crop(self.feature[index], self.label[index], self.crop_size[0], self.crop_size[1])
        return (self.tsf(feature), label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.feature)



if __name__ == '__main__':
    # 展示几张图像
    voc_dir = 'E:\\毕设\\训练数据\\voc2012\\VOCdevkit\\VOC2012'
    # train_features, train_labels = read_images(root=voc_dir, is_train=True, max_num=10)
    # n = 5
    # images = train_features[:n] + train_labels[:n]
    # show_images(images, num_rows=2, num_cols=n, scale=2)
    #
    colormap2label = torch.zeros(256 ** 3, dtype=torch.uint8) # 256 ** 3是因为这样可以让rgb每一位对应一个下标
    # 通过下面的循环，将voc2012中的20个类别在rgb色彩空间中标识出来
    for index, colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = index
    # y = label_indices(train_labels[0], colormap2label)
    # print(y[110:120, 215:274])

    # images = []
    # n = 5
    # for _ in range(n):
    #     # 裁剪200 * 300大小的子图
    #     images += rand_crop(train_features[0], train_labels[0], 200, 300)
    # # 从0开始，以2为步长来进行切片，再从1开始，以2为步长进行切片，这样就实现了将feature和label上下显示比较
    # show_images(images[::2] + images[1::2], 2, n)

    batch_size = 64
    crop_size = (320,480)
    max_num = 20000

    voc_train = VOCSegDataset(True, crop_size, voc_dir, colormap2label, max_num)
    voc_test = VOCSegDataset(False, crop_size, voc_dir, colormap2label, max_num)

    num_workers = 0 if sys.platform.startswith('win') else 4
    train_iter = torch.utils.data.DataLoader(voc_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(voc_test, batch_size=batch_size, drop_last=True, num_workers=num_workers)
    dataloaders = {'train': train_iter, 'test': test_iter}
    dataset_sizes = {'train': len(voc_train), 'test': len(voc_test)}
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = len(VOC_CLASSES)
    # 加载基于ImageNet的Resnet18训练的模型
    model_ft = torchvision.models.resnet18(pretrained=True) # true表明要加载训练好的参数

    # 关闭模型的梯度更新功能，也就是只使用它作为一个特征提取器
    for param in model_ft.parameters():
        param.requires_grad = False

    # print(model_ft)
    # model_ft = nn.Sequential(*list(model_ft.children())[:-2]).to(device)
    # x = torch.randn((2,3,320,480), device=device)
    # print(model_ft(x).size())

    model_ft = nn.Sequential(*list(model_ft.children())[:-2],   # 去掉最后两层全连接层
                             nn.Conv2d(512, num_classes, kernel_size= 1),
                             nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, padding=16, stride=32)).to(device)






