from torch.utils.data import DataLoader
from util.config.config_reader import ConfigReader, ConfigFileType
from util.data.segmentaion_dataset import *
from model.segformer.segformer import *
import torch.optim as optim
import torch.nn.functional as F

train_dataset = VOCSegmentationDataset(
    'E:\\毕设\\Cross_Domain_Few_Shot_Segmentation_System\\data\\train_data\\VOCdevkit\\VOC2012\\JPEGImages',
    'E:\\毕设\\Cross_Domain_Few_Shot_Segmentation_System\\data\\train_data\\VOCdevkit\\VOC2012\\SegmentationClass',
    'E:\\毕设\\Cross_Domain_Few_Shot_Segmentation_System\\data\\train_data\\VOCdevkit\\VOC2012\\ImageSets\\Segmentation\\train.txt',
    transform=transform
)
val_dataset = VOCSegmentationDataset(
    'E:\\毕设\\Cross_Domain_Few_Shot_Segmentation_System\\data\\train_data\\VOCdevkit\\VOC2012\\PEGImages',
    'E:\\毕设\\Cross_Domain_Few_Shot_Segmentation_System\\data\\train_data\\VOCdevkit\\VOC2012\\SegmentationClass',
    'E:\\毕设\\Cross_Domain_Few_Shot_Segmentation_System\\data\\train_data\\VOCdevkit\\VOC2012\\ImageSets\\Segmentation\\val.txt',
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

    return running_loss / len(val_loader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 21  # VOC 2012数据集有21个类别（包括背景）
    model = SegFormer(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'segformer_voc2012_model.pth')

if __name__ == '__main__':
    main()



