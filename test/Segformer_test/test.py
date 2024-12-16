from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from util.data.segmentaion_dataset import *
from util.tools.utils import seg_dataset_collate, worker_init_fn


def CE_Loss(inputs, target, cls_weights, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss

if __name__ == '__main__':
    train_lines = None
    val_lines = None
    input_shape=[512,512]
    num_classes=21
    dataset_path = 'E:\\毕设\\Cross_Domain_Few_Shot_Segmentation_System\\data\\train_data\\VOCdevkit\\VOC2012'
    workers = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open("E:\\毕设\\Cross_Domain_Few_Shot_Segmentation_System\\data\\train_data\\VOCdevkit\\VOC2012\\ImageSets\\Segmentation\\train.txt", encoding="utf-8", mode='r') as f:
        train_lines = f.readlines()
    with open("E:\\毕设\\Cross_Domain_Few_Shot_Segmentation_System\\data\\train_data\\VOCdevkit\\VOC2012\\ImageSets\\Segmentation\\val.txt", encoding="utf-8", mode='r') as f:
        val_lines = f.readlines()

    train_dataset = SegmentationDataset(train_lines, input_shape, num_classes, True, dataset_path)
    val_dataset = SegmentationDataset(val_lines, input_shape, num_classes, False, dataset_path)

    gen = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=workers, pin_memory=True,
                     drop_last=True, collate_fn=seg_dataset_collate, sampler=None,
                     worker_init_fn=partial(worker_init_fn, rank=0, seed=0))
    gen_val = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=workers, pin_memory=True,
                         drop_last=True, collate_fn=seg_dataset_collate, sampler=None,
                         worker_init_fn=partial(worker_init_fn, rank=0, seed=0))

    for iter, batch in enumerate(gen_val):
        test1 = iter
        tets2 = batch


    # model = SegFormer(num_classes=21,phi='b0',pretrained=False)
    # model_dict = model.state_dict()
    # pretrained_dict = torch.load("E:\\毕设\\Cross_Domain_Few_Shot_Segmentation_System\\data\\weights\\segformer_b0_backbone_weights.pth", map_location=device)
    # load_key, no_load_key, temp_dict = [],[],[]
    # for k, v in pretrained_dict.items():
    #     if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
    #         temp_dict[k] = v
    #         load_key.append(k)
    #     else:
    #         no_load_key.append(k)
    # model_dict.update(temp_dict)
    # model.load_state_dict(model_dict)
    #
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # cls_weights = np.ones([num_classes], np.float32)
    #
    #
    # model.train()
    # model.to(device)
    # epoch = 10
    # running_loss = 0.0
    # for iter, batch in enumerate(gen):
    #     imgs, pngs, labels = batch
    #     with torch.no_grad():
    #         imgs = imgs.to(device)
    #         pngs = pngs.to(device)
    #         labels = labels.to(device)
    #         weights = torch.from_numpy(cls_weights)
    #         weights = weights.to(device)
    #     optimizer.zero_grad()
    #     outputs = model(imgs)
    #     loss = CE_Loss(inputs=outputs, target=pngs, cls_weights=weights, num_classes= num_classes)
    #     loss.backward()
    #     # 优化
    #     optimizer.step()
    #     # 打印统计信息
    #     running_loss += loss.item()
    #     if iter % 10 == 9:  # 每100个batch打印一次
    #         print(f'[Epoch {epoch + 1}, Batch {iter + 1}] loss: {running_loss / 100:.3f}')
    #         running_loss = 0.0
    # print('Finished Training')