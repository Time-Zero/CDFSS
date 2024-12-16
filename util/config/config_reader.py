import os

import json5

from util.tools.singleton import singleton


@singleton
class ConfigReader:
    def __init__(self):
        self.config_path = "E:\\毕设\\Cross_Domain_Few_Shot_Segmentation_System\\config\\config.json5"
        self.data = None
        self.read_config()

    def read_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError('配置文件不存在，请检查文件路径')

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()

        try:
            self.data = json5.loads(config_content)
        except Exception as e:
            print(f'配置文件解析失败: {e}')

    def get_dataset_path(self):
        return self.data['train_data']['dataset_path']

    def get_num_classes(self):
        return self.data['train_data']['num_classes']

    def need_annotation(self):
        return self.data['train_data']['annotation']['need_annotation']

    def annotation_percent(self):
        train_data_per = self.data['train_data']['annotation']['train_dataset_per']
        val_data_per = self.data['train_data']['annotation']['val_dataset_per']
        test_data_per = self.data['train_data']['annotation']['test_dataset_per']

        if train_data_per + val_data_per + test_data_per != 1.:
            raise ValueError('train,val,test数据比例划分错误，三者之和应该为1')

        return train_data_per, val_data_per, test_data_per

    def auto_colored(self):
        return self.data['train_data']['colored']['auto_colored']

    def get_color_map(self):
        return self.data['train_data']['colored']['color_map']

    def get_num_workers(self):
        return self.data['model_train']['num_workers']

    def fp16_enable(self):
        return self.data['model_train']['fp16']

    def get_random_seed(self):
        return self.data['model_train']['random_seed']

    def get_input_shape(self):
        return self.data['model_train']['input_shape']

    def get_backbone(self):
        return self.data['model_train']['backbone']

    def get_pretrained_param(self):
        pretrained = self.data['model_train']['pretrained']
        backbone_pretrained = self.data['model_train']['backbone_pretrained']

        return pretrained, backbone_pretrained

    def get_backbone_weight_path(self):
        return self.data['model_train']['backbone_weight_path']

    def get_model_weight_path(self):
        return self.data['model_train']['model_weight_path']

    def get_model_save_path(self):
        return self.data['model_train']['model_save_path']

    def is_freeze_train(self):
        return self.data['model_train']['freeze_train']

    def get_epoch_param(self):
        init_epoch = self.data['model_train']['init_epoch']
        freeze_epoch = self.data['model_train']['freeze_epoch']
        unfreeze_epoch = self.data['model_train']['unfreeze_epoch']
        freeze_batch_size = self.data['model_train']['freeze_batch_size']
        unfreeze_batch_size = self.data['model_train']['unfreeze_batch_size']

        return init_epoch, freeze_epoch, unfreeze_epoch, freeze_batch_size, unfreeze_batch_size

    def get_lr_param(self):
        init_lr = self.data['model_train']['init_lr']
        min_lr_rate = self.data['model_train']['min_lr_rate']

        min_lr = init_lr * min_lr_rate

        return init_lr, min_lr

    def get_optimizer_param(self):
        optimizer_type = self.data['model_train']['optimizer_type']
        momentum = self.data['model_train']['momentum']

        return optimizer_type, momentum

    def get_weight_decay(self):
        return self.data['model_train']['weight_decay']

    def get_lr_decay_type(self):
        return self.data['model_train']['lr_decay_type']

    def dice_loss_enable(self):
        return self.data['model_train']['dice_loss']

    def focal_loss_enable(self):
        return self.data['model_train']['focal_loss']

    def get_cuda_enable(self):
        return self.data['base_param']['cuda_enable']

    def get_val_epoch(self):
        return self.data['model_train']['eval_epoch']

    def get_predict_cuda_enable(self):
        return self.data['predict_param']['cuda_enable']

    def get_predict_weight_path(self):
        return self.data['predict_param']['model_weight_path']

    def get_predict_num_classes(self):
        return self.data['predict_param']['num_classes']

    def get_predict_backbone(self):
        return self.data['predict_param']['backbone']

    def get_predict_input_shape(self):
        return self.data['predict_param']['input_shape']

    def get_predict_mix_type(self):
        return self.data['predict_param']['mix_type']

    def get_predict_mode(self):
        return self.data['predict_param']['mode']

    def is_count_pixel(self):
        return self.data['predict_param']['count']

    def get_classes_name(self):
        return self.data['predict_param']['classes_name']

if __name__ == '__main__':
    config = ConfigReader()
    print(config.get_dataset_path())
    print(config.get_num_classes())
    print(config.need_annotation())
    print(config.annotation_percent())
    print(config.auto_colored())
    print(config.get_color_map())
    print(config.get_num_workers())
    print(config.fp16_enable())
    print(config.get_random_seed())
    print(config.get_input_shape())
    print(config.get_backbone())
    print(config.get_pretrained_param())
    print(config.get_backbone_weight_path())
    print(config.get_model_weight_path())
    print(config.get_model_save_path())
    print(config.is_freeze_train())
    print(config.get_epoch_param())
    print(config.get_lr_param())
    print(config.get_optimizer_param())
    print(config.get_weight_decay())
    print(config.get_lr_decay_type())
    print(config.dice_loss_enable())
    print(config.focal_loss_enable())
    print(config.get_cuda_enable())
    print(config.get_val_epoch())
    print(config.get_predict_cuda_enable())
    print(config.get_predict_weight_path())
    print(config.get_predict_num_classes())
    print(config.get_predict_backbone())
    print(config.get_predict_input_shape())
    print(config.get_predict_mix_type())
    print(config.get_predict_mode())
    print(config.is_count_pixel())
    print(config.get_classes_name())
