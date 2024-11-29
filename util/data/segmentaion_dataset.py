import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class VOCSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_list_file, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        with open(image_list_file, 'r', encoding='utf-8') as f:
            self.image_files = [line.strip() for line in f]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx] + '.jpg')
        mask_path = os.path.join(self.mask_dir, self.image_files[idx] + '.png')

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = np.array(mask)
        mask = torch.from_numpy(mask).long()

        return image, mask

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])


