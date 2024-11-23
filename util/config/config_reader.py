import json
import os
from util.tools.singleton import *

@singleton
class ConfigReader:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config_data_dict = None

        if not os.path.exists(self.config_path):
            print("Config File Not Found, Please check your config file")
            exit(1)

        self.read_config()

    def read_config(self):
        """
        读取配置文件并把配置文件存放到self.config_data_dict中
        :return:
        """
        with open(self.config_path, 'r', encoding= 'utf-8') as json_file:
            self.config_data_dict = json.load(json_file)

    def get_config(self, key: str):
        """
        通过key获取对应的配置参数
        :param key: 参数名
        :return: 参数内容
        """
        if key not in self.config_data_dict:
            return None
        return self.config_data_dict[key]

if __name__ == '__main__':
    s1 = ConfigReader("E:\毕设\Cross_Domain_Few_Shot_Segmentation_System\config\config.json")
    print(s1.get_config("dataset")['path'])
