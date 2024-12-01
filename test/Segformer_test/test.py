from functools import partial
from torch.utils.data import DataLoader
from util.tools.utils import seg_dataset_collate, worker_init_fn
from util.data.segmentaion_dataset import *


if __name__ == '__main__':
    train_lines = None
    val_lines = None
    input_shape=[512,512]
    num_classes=21
    dataset_path = 'E:\\毕设\\Cross_Domain_Few_Shot_Segmentation_System\\data\\train_data\\VOCdevkit\\VOC2012'
    workers = 8

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

    for iter, batch in enumerate(gen):
        test = batch[0]
