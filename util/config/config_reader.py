import configparser
import os
from util.tools.singleton import *
from enum import Enum

class ConfigFileType(Enum):
    INT = 1
    FLOAT = 2
    STR = 3
    BOOL = 4

INIT_CONFIG_FILE_PATH="E:\毕设\Cross_Domain_Few_Shot_Segmentation_System\config\config.ini"

@singleton
class ConfigReader:
    def __init__(self):
        print("加载配置文件中")
        self.config_file_path = INIT_CONFIG_FILE_PATH
        self.config = None
        if not os.path.exists(self.config_file_path):
            raise ValueError("配置文件不存在，请检查配置文件路径")

        self.read_config()
        print("加载配置文件完成")

    def read_config(self):
        self.config = configparser.ConfigParser()
        self.config.read(INIT_CONFIG_FILE_PATH, encoding="utf-8")

    def get_config(self, section: str, option: str, type: ConfigFileType):
        match type:
            case ConfigFileType.INT:
                return self.config.getint(section, option)
            case ConfigFileType.FLOAT:
                return self.config.getfloat(section, option)
            case ConfigFileType.STR:
                return self.config.get(section, option)
            case ConfigFileType.BOOL:
                return self.config.getboolean(section, option)
            case _:
                raise ValueError("你需要返回的类型不合法")

if __name__ == '__main__':
    config = ConfigReader()
    print(config.get_config('train_data','path',ConfigFileType.STR))
    print(type(config.get_config('train_data','path',ConfigFileType.STR)))
    print(type(config.get_config('train_data','trainval_percent',ConfigFileType.INT)))
    print(type(config.get_config('train_data','need_annotation',ConfigFileType.BOOL)))
    print(type(config.get_config('train_data','train_percent',ConfigFileType.FLOAT)))


