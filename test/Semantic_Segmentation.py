import copy
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim, nn
import torchvision
from PIL import Image
from torch.optim import lr_scheduler
from torch.utils.data import dataloader
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

def bilinear_kernel(in_channels, out_channels, kernel_size):
    """
    使用双线性插值的上采样，用来初始化转置卷积层的卷积核
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :param kernel_size: 卷积核的F
    :return: 使用双线性插值的卷积核
    """
    # factor
    # 是卷积核的中心点。如果
    # kernel_size
    # 是奇数，中心点在卷积核的中间；如果
    # kernel_size
    # 是偶数，中心点在卷积核的中间偏下。
    factor = kernel_size // 2
    if kernel_size % 2 == 1:
        factor -= 1
    else:
        factor -= 0.5

    # 创建一个网格，用来生成滤波器
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - factor) / factor ) * (1 - abs(og[1] - factor) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    weight = torch.Tensor(weight)
    weight.requires_grad = True
    return weight

def train_model(model:nn.Module, criterion, optimizer, scheduler, num_epochs=20):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
            runing_loss = 0.0
            runing_correct = 0.0
            for inputs, labels in dataloader[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    logits = model(inputs)
                    loss = criterion(logits, labels.long())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                runing_loss += loss.item() * inputs.size(0)
                runing_correct += torch.sum((torch.argmax(logits.data,1)) == labels.data) / (480 * 320)

            epoch_loss = runing_loss / dataset_sizes[phase]
            epoch_acc = runing_correct.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    model.load_state_dict(best_model_wts)
    return model


def label2image(pred):
    colormap = torch.tensor(VOC_COLORMAP, device=device,dtype=torch.int)
    x = pred.long()
    return (colormap[x,:]).data.cpu().numpy()


def visualize_model(model:nn.Module, num_images=4):
    was_training = model.training
    model.eval()
    images_so_far = 0
    n, imgs = num_images, []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs, labels = inputs.to(device), labels.to(device) # [b,3,320,480]
            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=1) # [b,320,480]
            inputs_nd = (inputs*std+mean).permute(0,2,3,1)*255 # 记得要变回去哦

            for j in range(num_images):
                images_so_far += 1
                pred1 = label2image(pred[j]) # numpy.ndarray (320, 480, 3)
                imgs += [inputs_nd[j].data.int().cpu().numpy(), pred1, label2image(labels[j])]
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    # 我已经固定了每次只显示4张图了，大家可以自己修改
                    show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n)
                    return model.train(mode=was_training)


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

    # batch_size = 64
    # crop_size = (320,480)
    # max_num = 20000
    #
    # voc_train = VOCSegDataset(True, crop_size, voc_dir, colormap2label, max_num)
    # voc_test = VOCSegDataset(False, crop_size, voc_dir, colormap2label, max_num)
    #
    # num_workers = 0 if sys.platform.startswith('win') else 4
    # train_iter = torch.utils.data.DataLoader(voc_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    # test_iter = torch.utils.data.DataLoader(voc_test, batch_size=batch_size, drop_last=True, num_workers=num_workers)
    # dataloaders = {'train': train_iter, 'test': test_iter}
    # dataset_sizes = {'train': len(voc_train), 'test': len(voc_test)}
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = len(VOC_CLASSES)
    # 加载基于ImageNet的Resnet18训练的模型
    model_ft = torchvision.models.resnet18(pretrained=True) # true表明要加载训练好的参数

    # 关闭模型的梯度更新功能，也就是只使用它作为一个特征提取器
    for param in model_ft.parameters():
        param.requires_grad = False

    print(model_ft)
    model_ft = nn.Sequential(*list(model_ft.children())[:-2]).to(device)
    x = torch.randn((2,3,320,480), device=device)
    print(model_ft(x).size())

    # model_ft = nn.Sequential(*list(model_ft.children())[:-2],   # 去掉最后两层全连接层
    #                          nn.Conv2d(512, num_classes, kernel_size= 1),
    #                          nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, padding=16, stride=32)).to(device)
    #
    # nn.init.xavier_normal_(model_ft[-2].weight.data, gain = 1)      # 倒数第二层使用xavier随机初始化
    # model_ft[-1].weight.data = bilinear_kernel(num_classes, num_classes, 64).to(device)  # 最后一层使用双线性插值初始化
    #
    # epochs = 5
    # criteon = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model_ft.parameters(), lr=0.001, weight_decay=1e-4, momentum=0.9)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # model_ft = train_model(model_ft, criteon, optimizer, exp_lr_scheduler, num_epochs=epochs)
    #
    # mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1).to(device)
    # std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1).to(device)
    # visualize_model(model_ft)





