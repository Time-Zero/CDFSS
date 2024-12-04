import os
from functools import partial
from termcolor import colored
from torch.backends import cudnn
from torch.utils.data import DataLoader
from model.segformer.segformer import SegFormer
from util.config.config_reader import ConfigReader, ConfigFileType
from util.tools.utils import *
from torch.amp import GradScaler
from util.data.segmentaion_dataset import SegmentationDataset
from util.tools.utils_model import set_optimizer_lr, get_lr_scheduler, EnumOptimizer, optimizer_select, fit_one_epoch


def train_model():
    # 加载配置项
    config = ConfigReader()
    cuda_enable = config.get_config('model_train', 'cuda', ConfigFileType.BOOL)
    seed = config.get_config('model_train', 'seed', ConfigFileType.INT)
    pretrained = config.get_config('model_train', 'pretrained', ConfigFileType.BOOL)
    backbone_pretrained = config.get_config('model_train', 'backbone_pretrained', ConfigFileType.BOOL)
    backbone_weight_path = config.get_config('model_train', 'backbone_weight_path', ConfigFileType.STR)
    model_weight_path = config.get_config('model_train', 'model_weight_path', ConfigFileType.STR)
    num_classes = config.get_config('model_train', 'num_classes', ConfigFileType.INT) + 1
    feature_extraction_fun = config.get_config('model_train', 'feature_extraction_fun', ConfigFileType.STR)
    fp16 = config.get_config('model_train', 'fp16', ConfigFileType.BOOL)
    init_epoch = config.get_config('model_train', 'init_epoch', ConfigFileType.INT)
    freeze_epoch = config.get_config('model_train', 'freeze_epoch', ConfigFileType.INT)
    unfreeze_epoch = config.get_config('model_train', 'unfreeze_epoch', ConfigFileType.INT)
    freeze_train = config.get_config('model_train', 'freeze_train', ConfigFileType.BOOL)
    freeze_batch_size = config.get_config('model_train', 'freeze_batch_size', ConfigFileType.INT)
    unfreeze_batch_size = config.get_config('model_train', 'unfreeze_batch_size', ConfigFileType.INT)
    init_lr = config.get_config('model_train', 'init_lr', ConfigFileType.FLOAT)
    dataset_path = config.get_config('train_data', 'path', ConfigFileType.STR)
    optimizer_type = config.get_config('model_train', 'optimizer_type', ConfigFileType.STR)
    min_lr_rate = config.get_config('model_train', 'min_lr_rate', ConfigFileType.FLOAT)
    momentum = config.get_config('model_train', 'momentum', ConfigFileType.FLOAT)
    weight_decay = config.get_config('model_train', 'weight_decay', ConfigFileType.FLOAT)
    lr_decay_type = config.get_config('model_train', 'lr_decay_type', ConfigFileType.STR)
    input_shape_str = config.get_config('model_train', 'input_shape', ConfigFileType.STR)
    input_shape = list(map(int, input_shape_str.split(',')))
    num_workers = config.get_config('model_train', 'num_workers', ConfigFileType.INT)
    dice_loss = config.get_config('model_train', 'dice_loss', ConfigFileType.BOOL)
    focal_loss = config.get_config('model_train', 'focal_loss', ConfigFileType.BOOL)

    cls_weights = np.ones([num_classes], np.float32)

    # 读取train_lines和val_lines
    train_path = os.path.join(dataset_path, 'ImageSets/Segmentation/train.txt')
    val_path = os.path.join(dataset_path, 'ImageSets/Segmentation/val.txt')
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise ValueError('train.txt或val.txt不存在，请检查项目文件结构，或运行数据标注模式生成')

    with open(train_path, 'r', encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_path, 'r', encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)            # 训练数据总数
    num_val = len(val_lines)                # 测试数据总数

    # 初始化随机数种子
    random_seed_init(seed)

    # 选择计算设备
    device = torch.device('cpu')
    if cuda_enable:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            print(colored('警告：gpu不可用，正在使用cpu运算！！！', 'red'))

    # 预训练权重导入
    if pretrained:
        # 如果导入预训练权重
        if backbone_pretrained:
            # 仅加载主干网络预训练权重
            if not os.path.exists(backbone_weight_path):
                raise ValueError('主干网络预训练权重文件不存在，请检查')

            model = SegFormer(num_classes=num_classes, phi=feature_extraction_fun, pretrained=backbone_pretrained, backbone_weight_path=backbone_weight_path)
        else:
            # 加载全局预训练权重
            model = SegFormer(num_classes=num_classes, phi=feature_extraction_fun, pretrained=backbone_pretrained)
            model_dict = model.state_dict()
            load_key, no_load_key, temp_dict = pretrained_weight_load(model_dict=model_dict, weight_path=model_weight_path, device=device)

            model_dict.update(temp_dict)
            model.load_state_dict(model_dict)

            print('\n')
            print(colored(f"成功加载权值key：{load_key[:500]}", "green"))
            print(colored(f"成功加载key的数量为：{len(load_key)}", "green"))
            print('\n')
            print(colored(f"加载失败权值key：{no_load_key[:500]}", "yellow"))
            print(colored(f"加载失败key的数量为：{len(no_load_key)}", "yellow"))
    else:
        # 如果不导入预训练权重
        model = SegFormer(num_classes=num_classes, phi=feature_extraction_fun, pretrained=False)

    # 启用混合精度
    if fp16:
        scaler = GradScaler()
    else:
        scaler = None


    model_train = model.train()
    cudnn.benchmark = True
    model_train = model_train.to(device)

    # 如果是冻结训练，冻结主干网络
    if freeze_train:
        for param in model.backbone.parameters():
            param.requires_grad = False

    # 如果是冻结训练，则batch_size设置为freeze_batch_size,否则使用非冻结训练batch_size
    batch_size = freeze_batch_size if freeze_train else unfreeze_batch_size

    # 根据当前batch_size，自适应调整学习率
    optimizer_type = {'adam': EnumOptimizer.ADAM,
                      'adamw': EnumOptimizer.ADAMW,
                      'sgd': EnumOptimizer.SGD}[optimizer_type]
    min_lr = init_lr * min_lr_rate
    init_lr_fit, min_lr_fit = calculate_lf_fit(nbs = 16, optimizer_type=optimizer_type, batch_size=batch_size, init_lr=init_lr, min_lr=min_lr)

    # 选择优化器
    optimizer = optimizer_select(optimizer_type=optimizer_type, model=model, init_lr_fit=init_lr_fit, momentum=momentum, weight_decay=weight_decay)

    # 获得学习率下降公式
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, init_lr_fit, min_lr_fit, unfreeze_epoch)

    # 计算每一个epoch长度
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError('数据集小于batch_size，请扩充数据集')

    train_dataset = SegmentationDataset(train_lines, input_shape, num_classes, True, dataset_path)
    val_dataset = SegmentationDataset(val_lines, input_shape, num_classes, False, dataset_path)

    train_sampler = None
    val_sampler = None
    shuffle = True

    gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                     drop_last=True, collate_fn=seg_dataset_collate, sampler=train_sampler,
                     worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
    gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=seg_dataset_collate, sampler=val_sampler,
                         worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

    #TODO: 修复gen_val的enumerate报错问题
    for iter, batch in enumerate(gen_val):
        test1 = iter
        test2 = batch

    unfreeze_flag = False       # 标志：防止从冻结epoch到解冻epoch后batch_size被多次设置
    for epoch in range(init_epoch, unfreeze_epoch):

        if epoch >= freeze_epoch and not unfreeze_flag and freeze_train:
        # 如果是冻结学习，并且到了解冻阶段，则解冻模型

            # 赋值新的batch_size
            batch_size = unfreeze_batch_size

            # 计算每一个epoch应该计算的数据
            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size

            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

            # 由于batch_size在冻结阶段和解冻阶段是不一致的，所以需要再一次计算学习率
            init_lr_fit, min_lr_fit = calculate_lf_fit(nbs=16, optimizer_type=optimizer_type, batch_size=batch_size,
                                                       init_lr=init_lr, min_lr=min_lr)

            lr_scheduler_func = get_lr_scheduler(lr_decay_type, init_lr_fit, min_lr_fit, unfreeze_epoch)

            # 解冻模型
            for param in model.backbone.parameters():
                param.requires_grad = True


            gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=seg_dataset_collate, sampler=train_sampler,
                             worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
            gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=seg_dataset_collate, sampler=val_sampler,
                                 worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

            unfreeze_flag = True

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        fit_one_epoch(model_train=model_train, model=model, optimizer=optimizer, num_classes=num_classes,
                      cur_epoch=epoch, epoch_step=epoch_step, epoch_step_val=epoch_step_val, gen=gen,
                      gen_val=gen_val, total_epoch=unfreeze_epoch, cuda_enable=cuda_enable,
                      focal_loss_flag=focal_loss, dice_loss_flag=dice_loss, cls_weights=cls_weights,
                      fp16=fp16, scaler=scaler)

