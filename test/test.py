# from utils.config_reader import ConfigReader
#
# if __name__ == '__main__':
#     config = ConfigReader()
#     config.set_num_classes(2)
#     config.mp_dump_config()
#     config.mp_reload_config()
#     print(config.get_num_classes())
from PIL import Image
import numpy as np

if __name__ == "__main__":
    file_path = 'E:\\毕设\\CDFSS\\test\\temp\\img1.png'
    img = Image.open(file_path)
    img_np = np.array(img)
    print(img_np)