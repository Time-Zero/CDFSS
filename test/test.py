from utils.config_reader import ConfigReader

if __name__ == '__main__':
    config = ConfigReader()
    config.set_num_classes(2)
    config.mp_dump_config()
    config.mp_reload_config()
    print(config.get_num_classes())