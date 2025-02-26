import os

import json5

from utils.singleton import singleton

@singleton
class ConfigReader:

    def __init__(self):
        self._cuda_enable = None
        self._pred_cuda = None
        self._is_predict = None
        self._num_classes = None
        self._name_classes = None
        self._dataset_path = None
        self.file_path = "./config/config.json5"
        self.temp_path = "./tmp"
        self.data = None

        self._read_config()

    def mp_dump_config(self):
        """
        和mp_reload_config()组合使用，实现多进程参数同步
        配置的保存
        :return:
        """
        data_str = json5.dumps(self.data, indent=4, quote_keys=True)

        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)

        temp_file_path = os.path.join(self.temp_path, "temp.json5")
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            f.write(data_str)

    def mp_reload_config(self):
        """
        和mp_dump_config()组合使用，实现多进程参数同步
        配置的加载
        :return:
        """
        self.set_conf_path(os.path.join(self.temp_path, "temp.json5"))

    def _read_config(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError('配置文件不存在，请检查文件路径')

        with open(self.file_path, 'r', encoding='utf-8') as f:
            config_content = f.read()

        try:
            self.data = json5.loads(config_content)
        except Exception as e:
            print(f'配置文件解析失败，请检查配置文件语法: {e}')

        # ------------------------random_seed----------------------
        self._random_seed = self.data['base']['random_seed']
        self._is_train = self.data['base']['process_param']['train']
        self._is_predict = self.data['base']['process_param']['predict']

        # --------------------------dataset参数---------------------------------------
        self._dataset_path = self.data['dataset']['path']
        self._preprocess_enable = self.data['dataset']['preprocess_param']['enable']
        self._preprocess_color = self.data['dataset']['preprocess_param']['need_color']
        self._need_divide = self.data['dataset']['preprocess_param']['divide']['need_divide']
        self._divide_per = self.data['dataset']['preprocess_param']['divide']['divide_percent']
        self._need_crop = self.data['dataset']['preprocess_param']['random_crop']['need_crop']
        self._crop_size = self.data['dataset']['preprocess_param']['random_crop']['size']
        self._auto_color = self.data['dataset']['color']['auto_color']
        self._color_map = self.data['dataset']['color']['colors_map']
        self._num_classes = self.data['dataset']['num_classes']
        self._cls_weight_enable = self.data['dataset']['cls_weights']['enable']
        self._name_classes = self.data['dataset']['name_classes']
        self._is_auto_get_num_name = self.data['dataset']['auto_get_num_name']

        # ----------------------------模型参数-------------------------------------
        self._num_workers = self.data['model']['num_workers']
        self._phi = self.data['model']['phi']
        self._input_size = self.data['model']['input_size']

        # ------------------------------train------------------------------------------
        self._fp16 = self.data['train']['fp16']
        self._cuda_enable = self.data['train']['cuda']
        self._focal_loss = self.data['train']['focal_loss']
        self._dice_loss = self.data['train']['dice_loss']
        self._eval_freq = self.data['train']['eval_freq']
        self._pretrained = self.data['train']['pretrained']['enable']
        self._pretrained_weight = self.data['train']['pretrained']['weight']
        self._pretrained_weight_path = self.data['train']['pretrained']['weight_path']
        self._freeze_train_enable = self.data['train']['freeze_train']['enable']
        self._init_epoch = self.data['train']['freeze_train']['init_epoch']
        self._freeze_epoch = self.data['train']['freeze_train']['freeze_epoch']
        self._freeze_batch_size = self.data['train']['freeze_train']['freeze_batch_size']
        self._unfreeze_epoch = self.data['train']['freeze_train']['unfreeze_epoch']
        self._unfreeze_batch_size = self.data['train']['freeze_train']['unfreeze_batch_size']
        self._init_lr = self.data['train']['lr_param']['init_lr']
        self._min_lr_ratio = self.data['train']['lr_param']['min_lr_ratio']
        self._optimizer_type = self.data['train']['optimizer_param']['optimizer_type']
        self._momentum = self.data['train']['optimizer_param']['momentum']
        self._weight_decay = self.data['train']['optimizer_param']['weight_decay']
        self._lr_decay_type = self.data['train']['lr_param']['lr_decay_type']
        self._log_dir = self.data['train']['log_dir']
        self._weight_save_freq = self.data['train']["weight_save_param"]['weight_save_freq']
        self._weight_save_path = self.data['train']["weight_save_param"]['weight_save_path']

        # ---------------------------------------预测----------------------------------
        self._pred_res_save_path = self.data['pred']['save_path']
        self._pred_model_path = self.data['pred']['model_path']
        self._pred_cuda = self.data['pred']['cuda']
        self._pre_out_color = self.data['pred']['out_color']

    def pre_out_color(self):
        return self._pre_out_color

    def preprocess_color(self):
        return self._preprocess_color

    def is_preprocess(self):
        return self._preprocess_enable

    def is_auto_get_num_name(self):
        return self._is_auto_get_num_name

    def set_conf_path(self, conf_path):
        self.file_path = os.path.normpath(conf_path)
        self._read_config()

    def is_train(self):
        return self._is_train

    def is_predict(self):
        return self._is_predict

    def get_name_classes(self):
        return self._name_classes

    def set_name_classes(self, name_classes):
        self.data['dataset']['name_classes'] = name_classes
        self._name_classes = name_classes
    
    def set_pred_cuda_enable(self, pred_cuda_enable):
        self.data['pred']['cuda'] = pred_cuda_enable
        self._pred_cuda = pred_cuda_enable

    def pred_cuda_enable(self):
        return self._pred_cuda

    def get_model_path(self):
        return os.path.normpath(self._pred_model_path)

    def get_pred_res_save_path(self):
        return os.path.normpath(self._pred_res_save_path)

    def get_weight_save_param(self):
        return self._weight_save_freq, os.path.normpath(self._weight_save_path)

    def get_log_dir(self):
        return self._log_dir

    def focal_loss_enable(self):
        return self._focal_loss

    def dice_loss_enable(self):
        return self._dice_loss

    def eval_freq(self):
        return self._eval_freq

    def get_input_size(self):
        return self._input_size

    def get_lr_decay_type(self):
        if self._lr_decay_type not in ['step', 'cos']:
            raise ValueError(f'你当前选择学习率下降方式不支持: {self._lr_decay_type}')
        return self._lr_decay_type

    def get_optimizer_param(self):
        if self._optimizer_type not in ['adamw', 'adam', 'sgd']:
            raise ValueError(f'你选择优化器: "{self._optimizer_type}" 不支持')
        return self._optimizer_type, self._momentum, self._weight_decay

    def get_min_lr(self):
        return self._min_lr_ratio * self._init_lr

    def get_init_lr(self):
        return self._init_lr

    def get_epoch_param(self):
        return (self._init_epoch, self._freeze_epoch, self._freeze_batch_size,
                self._unfreeze_epoch, self._unfreeze_batch_size)

    def freeze_train_enable(self):
        return self._freeze_train_enable

    def get_phi(self):
        return self._phi

    def get_pretrained_param(self):
        if self._pretrained_weight not in ['full', 'backbone']:
            raise ValueError('pretrained weight 必须为 full 或 backbone')

        return self._pretrained_weight, os.path.normpath(self._pretrained_weight_path)

    def pretrained_enable(self):
        return self._pretrained

    def set_cuda_enable(self,value: bool):
        self.data['train']['cuda'] = value
        self._cuda_enable = value

    def cuda_enable(self):
        return self._cuda_enable

    def need_divide(self):
        return self._need_divide

    def get_num_workers(self):
        return self._num_workers

    def get_fp16(self):
        return self._fp16

    def get_dataset_path(self):
        return os.path.normpath(self._dataset_path)

    def set_dataset_path(self, dataset_path):
        self.data['dataset']['path'] = os.path.normpath(dataset_path)
        self._dataset_path = dataset_path

    def get_divide_percent(self):
        return self._divide_per

    def need_crop(self):
        return self._need_crop

    def get_crop_size(self):
        return self._crop_size

    def get_auto_color(self):
        return self._auto_color

    def get_color_map(self):
        return self._color_map

    def set_num_classes(self, num_classes):
        self.data['dataset']['num_classes'] = num_classes
        self._num_classes = num_classes

    def get_num_classes(self):
        return self._num_classes

    def cls_weight_enable(self):
        return self._cls_weight_enable

    def get_random_seed(self):
        return self._random_seed
