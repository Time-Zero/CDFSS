from utils.utils_dataset import *

def test_dataset_divide():
    dataset_path = "E:\毕设\Cross_Domain_Few_Shot_Segmentation_System\data\dataset\FloodNet_Voc_Format"
    dataset_divide(dataset_path, 0.7, 0.15, 0.15)

def test_image_crop():
    dataset_path = "E:\毕设\Cross_Domain_Few_Shot_Segmentation_System\data\dataset\FloodNet_Voc_Format"
    feature = Image.open(os.path.join(dataset_path, "JPEGImages/6279.jpg"))
    label = Image.open(os.path.join(dataset_path, "SegmentationClass/6279.png"))

    image_crop(feature, label, [768,768])
    feature.show()
    label.show()

if __name__ == '__main__':
    # test_dataset_divide()
    # test_image_crop()
    pass