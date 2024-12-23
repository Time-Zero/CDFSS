import os

import json5

from utils.singleton import singleton


@singleton
class ConfigReader:
    def __init__(self):
        self.__dataset_path = None
        self.file_path = "../config/config.json5"
        self.data = None
        self.dataset_path = None

        self.__read_config()

    def __read_config(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError('配置文件不存在，请检查文件路径')

        with open(self.file_path, 'r', encoding='utf-8') as f:
            config_content = f.read()

        try:
            self.data = json5.loads(config_content)
        except Exception as e:
            print('配置文件解析失败，请检查配置文件语法: {e}')

        # ------------------------random_seed----------------------
        self.__random_seed = self.data['base']['random_seed']

        # --------------------------dataset参数---------------------------------------
        self.__dataset_path = self.data['dataset']['path']
        self.__need_divide = self.data['dataset']['preprocess_param']['divide']['need_divide']
        self.__divide_per = self.data['dataset']['preprocess_param']['divide']['divide_percent']
        self.__need_crop = self.data['dataset']['preprocess_param']['random_crop']['need_crop']
        self.__crop_size = self.data['dataset']['preprocess_param']['random_crop']['size']
        self.__auto_color = self.data['dataset']['color']['auto_color']
        self.__color_map = self.data['dataset']['color']['colors_map']
        self.__need_color = self.data['dataset']['color']['need_color']
        self.__num_classes = self.data['dataset']['num_classes']
        self.__cls_weight_enable = self.data['dataset']['cls_weight']['enable']
        self.__cls_weight = self.data['dataset']['cls_weight']['cls_weight']

        # ----------------------------模型参数-------------------------------------
        self.__num_workers = self.data['model']['num_workers']
        self.__fp16 = self.data['model']['fp16']
        self.__cuda_enable = self.data['model']['cuda_param']['cuda']
        self.__cuda_mode = self.data['model']['cuda_param']['mode']
        self.__cuda_visible_device = self.data['model']['cuda_param']['visible_device']
        self.__cuda_master_gpu = self.data['model']['cuda_param']['master_gpu']
        self.__phi = self.data['model']['phi']
        self.__input_size = self.data['model']['input_size']

        # ------------------------------train------------------------------------------
        self.__focal_loss = self.data['train']['focal_loss']
        self.__dice_loss = self.data['train']['dice_loss']
        self.__eval_freq = self.data['train']['eval_freq']
        self.__pretrained = self.data['train']['pretrained']['enable']
        self.__pretrained_weight = self.data['train']['pretrained']['weight']
        self.__pretrained_weight_path = self.data['train']['pretrained']['weight_path']
        self.__freeze_train_enable = self.data['train']['freeze_train']['enable']
        self.__init_epoch = self.data['train']['freeze_train']['init_epoch']
        self.__freeze_epoch = self.data['train']['freeze_train']['freeze_epoch']
        self.__freeze_batch_size = self.data['train']['freeze_train']['freeze_batch_size']
        self.__unfreeze_epoch = self.data['train']['freeze_train']['unfreeze_epoch']
        self.__unfreeze_batch_size = self.data['train']['freeze_train']['unfreeze_batch_size']
        self.__init_lr = self.data['train']['lr_param']['init_lr']
        self.__min_lr_ratio = self.data['train']['lr_param']['min_lr_ratio']
        self.__optimizer_type = self.data['train']['optimizer_param']['optimizer_type']
        self.__momentum = self.data['train']['optimizer_param']['momentum']
        self.__weight_decay = self.data['train']['optimizer_param']['weight_decay']
        self.__lr_decay_type = self.data['train']['lr_param']['lr_decay_type']
        self.__log_dir = self.data['train']['log_dir']
        self.__weight_save_freq = self.data['train']["weight_save_param"]['weight_save_freq']
        self.__weight_save_path = self.data['train']["weight_save_param"]['weight_save_path']

    def get_weight_save_param(self):
        return self.__weight_save_freq, self.__weight_save_path

    def get_log_dir(self):
        return self.__log_dir

    def focal_loss_enable(self):
        return self.__focal_loss

    def dice_loss_enable(self):
        return self.__dice_loss

    def eval_freq(self):
        return self.__eval_freq

    def get_input_size(self):
        return self.__input_size

    def get_lr_decay_type(self):
        if self.__lr_decay_type not in ['step','cos']:
            raise ValueError(f'你当前选择学习率下降方式不支持: {self.__lr_decay_type}')
        return self.__lr_decay_type

    def get_optimizer_param(self):
        if self.__optimizer_type not in ['adamw', 'adam', 'sgd']:
            raise ValueError(f'你选择优化器: "{self.__optimizer_type}" 不支持')
        return self.__optimizer_type, self.__momentum, self.__weight_decay

    def get_min_lr(self):
        return self.__min_lr_ratio * self.__init_lr

    def get_init_lr(self):
        return self.__init_lr

    def get_epoch_param(self):
        return (self.__init_epoch, self.__freeze_epoch, self.__freeze_batch_size,
                self.__unfreeze_epoch, self.__unfreeze_batch_size)

    def freeze_train_enable(self):
        return self.__freeze_train_enable

    def get_phi(self):
        return self.__phi

    def get_pretrained_param(self):
        if self.__pretrained_weight not in ['full','backbone']:
            raise ValueError('pretrained weight 必须为 full 或 backbone')

        return self.__pretrained_weight, self.__pretrained_weight_path

    def pretrained_enable(self):
        return self.__pretrained

    def get_cuda_master_gpu(self):
        return self.__cuda_master_gpu

    def get_cuda_visible_gpus(self):
        # return ",".join(map(str, self.__cuda_visible_device))
        return self.__cuda_visible_device

    def get_cuda_mode(self):
        if self.__cuda_mode not in ['none','dp','ddp']:
            raise ValueError('cuda模式设置错误，请检查配置文件')
        return self.__cuda_mode

    def cuda_enable(self):
        return self.__cuda_enable

    def need_divide(self):
        return self.__need_divide

    def get_num_workers(self):
        return self.__num_workers

    def get_fp16(self):
        return self.__fp16

    def get_dataset_path(self):
        return self.__dataset_path

    def set_dataset_path(self, dataset_path):
        self.__dataset_path = dataset_path

    def get_divide_percent(self):
        return self.__divide_per

    def need_crop(self):
        return self.__need_crop

    def get_crop_size(self):
        return self.__crop_size

    def get_auto_color(self):
        return self.__auto_color

    def get_color_map(self):
        return self.__color_map

    def need_color(self):
        return self.__need_color

    def get_num_classes(self):
        return self.__num_classes

    def cls_weight_enable(self):
        return self.__cls_weight_enable

    def get_cls_weight(self) -> list:
        return self.__cls_weight

    def get_random_seed(self):
        return self.__random_seed