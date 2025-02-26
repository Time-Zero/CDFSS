# from utils.config_reader import ConfigReader
#
# if __name__ == '__main__':
#     config = ConfigReader()
#     config.set_num_classes(2)
#     config.mp_dump_config()
#     config.mp_reload_config()
#     print(config.get_num_classes())
# from utils.utils_model import compute_cls_weights_optimized
#
# if __name__ == "__main__":
#     # file_path = 'E:\\毕设\\CDFSS\\test\\temp\\img1.png'
#     # img = Image.open(file_path)
#     # img_np = np.array(img)
#     # print(img_np)
#     file_path = "E:\毕设\CDFSS\data\dataset\SS\CHASEDB1_VOC\SegmentationClass"
#
#     res = compute_cls_weights_optimized(file_path,2)
#     print(res)

import torch

if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())