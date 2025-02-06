import datetime
import os
import platform
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from colorama import Fore, Style
from torch import optim
from torch.amp import GradScaler
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from model.loss_history import LossHistory
from model.segformer import SegFormer
from model.segmentaion_dataset import SegmentationDataset
from utils.config_reader import ConfigReader
from utils.utils_model import random_seed_init, get_lr_scheduler, seg_dataset_collate, worker_init_fn, set_optimizer_lr, \
    fit_one_epoch


def train_controller():
    """
    训练控制模块，主要是为了使用mp.spawn来启用ddp
    :return:
    """
    config = ConfigReader()

    if not config.cuda_enable():
        train()
    else:
        cuda_mode = config.get_cuda_mode()
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, config.get_cuda_visible_gpus()))

        if cuda_mode == 'ddp':
            mp.spawn(train,
                     nprocs=torch.cuda.device_count(),
                     join=True)
        else:
            train()


def train(rank: int = 0):
    config = ConfigReader()
    cuda_enable = config.cuda_enable()
    cuda_mode = config.get_cuda_mode()

    if cuda_enable:
        devices_ids = config.get_cuda_visible_gpus()

    # ----------------------初始化全局随机数种子-------------------
    random_seed_init(config.get_random_seed())

    # ---------------------是否启用类别偏置权重来解决数据集不平衡问题----------------------
    num_classes = config.get_num_classes()
    if config.cls_weight_enable():
        cls_weight = np.array(config.get_cls_weight(), dtype=np.float32)
    else:
        cls_weight = np.ones(num_classes, dtype=np.float32)

    # --------------------cuda配置----------------------------
    if not cuda_enable:
        # cpu模式
        device = torch.device('cpu')

    else:
        # gpu模式

        if cuda_mode == 'none':
            device = torch.device('cuda')
        elif cuda_mode == 'dp':
            # dp模式
            device = torch.device('cuda')
        else:
            # ddp模式
            plat = platform.system()
            backend = 'gloo' if plat == 'Windows' else 'nccl'
            world_size = torch.cuda.device_count()
            addr = "localhost"
            port = 23456
            dist.init_process_group(backend=backend, world_size=world_size,
                                    rank=rank, init_method=f"tcp://{addr}:{port}?use_libuv=0")
            device = torch.device('cuda', rank)

            if rank == 0:
                print(Fore.BLUE + f"[{os.getpid()}] (rank = {rank}) 训练中...." + Style.RESET_ALL)
                print(Fore.BLUE + f"Gpu Device Count : {world_size}" + Style.RESET_ALL)

    # ------------------------------模型初始化（预训练权重加载）---------------------------
    num_classes = config.get_num_classes()
    phi = config.get_phi()
    if not config.pretrained_enable():
        if rank == 0:
            print(Fore.GREEN + '不使用预训练权重' + Style.RESET_ALL)
        # 不使用预训练权重
        model = SegFormer(num_classes=num_classes, phi=phi, pretrained=False, backbone_weight_path='')
    else:
        weight_mode, weight_path = config.get_pretrained_param()

        if weight_mode == 'backbone':
            if rank == 0:
                print(Fore.GREEN + f'主干网络将加载预训练权重, path: {weight_path}' + Style.RESET_ALL)
            # 主干网络使用预训练权重
            model = SegFormer(num_classes=num_classes, phi=phi, pretrained=True, backbone_weight_path=weight_path)
        else:
            # 全局使用预训练权重
            model = SegFormer(num_classes=num_classes, phi=phi, pretrained=False, backbone_weight_path='')

            if rank == 0:
                print(Fore.GREEN + f'全局网络将加载预训练权重, path: {weight_path}' + Style.RESET_ALL)

            model_dict = model.state_dict()
            pretrained_dict = torch.load(weight_path, map_location=device, weights_only=True)
            load_key, no_load_key, temp_dict = [], [], {}
            for k, v in pretrained_dict.items():
                if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                    temp_dict[k] = v
                    load_key.append(k)
                else:
                    no_load_key.append(k)
            model_dict.update(temp_dict)
            model.load_state_dict(model_dict)

            # 显示没有加载成功的权重
            if rank == 0:
                print(Fore.GREEN + f'加载成功的权重为: {str(load_key)[:500]}')
                print(f'加载成功的权重数量为: {len(load_key)}' + Style.RESET_ALL)
                print(Fore.YELLOW + f'加载失败的权重为: {str(no_load_key)[:500]}')
                print(f'加载失败的权重数量为: {len(no_load_key)}' + Style.RESET_ALL)
                print(Fore.BLUE + f'head有权重加载失败是正常的' + Style.RESET_ALL)

    # -------------------------------------- 启用tensorboard---------------------------------
    if rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(config.get_log_dir(), "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, config.get_input_size())
        print(Fore.YELLOW +
              f'TensorBoard日志路径为: {os.path.join(os.getcwd() , log_dir)}' + Style.RESET_ALL)
    else:
        loss_history = None

    # ------------------------------------- 是否启用fp16---------------------------------
    fp16 = config.get_fp16()
    if fp16:
        scaler = GradScaler()
    else:
        scaler = None

    # -----------------------------启用Sync_BatchNorm（将提高训练的一致性）----------------------
    model_train = model.train()

    if cuda_enable:
        if torch.cuda.device_count() > 1 and cuda_mode == 'ddp':
            model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
        else:
            print(Fore.YELLOW + 'Sync_Batchnorm没有启用，可能是由于单卡训练或非DDP模式' + Style.RESET_ALL)

        # ----------------------------分布式训练时将模型切分------------------------------------
        if cuda_mode == 'none':
            model_train.cuda()
        elif cuda_mode == 'dp':
            model_train = torch.nn.DataParallel(model_train, device_ids=devices_ids)
            cudnn.benchmark = True
            model_train.cuda()
        elif cuda_mode == 'ddp':
            model_train = model_train.cuda(rank)
            model_train = DDP(model_train, device_ids=[rank], find_unused_parameters=True)

    # --------------------------------读取数据集----------------------------------------
    dataset_path = config.get_dataset_path()
    if rank == 0:
        print(Fore.BLUE + f'加载训练数据集: {dataset_path}' + Style.RESET_ALL)
    with open(os.path.join(dataset_path, "ImageSets\\Segmentation\\train.txt"), 'r', encoding='utf-8') as f:
        train_lines = f.readlines()
        train_lines = [line.strip() for line in train_lines]
    with open(os.path.join(dataset_path, "ImageSets\\Segmentation\\val.txt"), 'r', encoding='utf-8') as f:
        val_lines = f.readlines()
        val_lines = [line.strip() for line in val_lines]
    num_train = len(train_lines)
    num_val = len(val_lines)

    if rank == 0:
        print(Fore.BLUE + '加载数据集成功' + Style.RESET_ALL, flush=True)

    optimizer_type, momentum, weight_decay = config.get_optimizer_param()
    init_lr = config.get_init_lr()
    min_lr = config.get_min_lr()
    # --------------------------------freeze_train配置------------------------
    init_epoch, freeze_epoch, freeze_batch_size, unfreeze_epoch, unfreeze_batch_size = config.get_epoch_param()
    unfreeze_flag = False
    freeze_train = config.freeze_train_enable()
    if freeze_train:
        for param in model.backbone.parameters():
            param.requires_grad = False

    batch_size = freeze_batch_size if config.freeze_train_enable() else unfreeze_batch_size

    # --------------------------------通过当前的batch_size计算学习率--------------------------------
    nbs = 16
    lr_limit_max = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
    lr_limit_min = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
    init_lr_fit = min(max(batch_size / nbs * init_lr, lr_limit_min), lr_limit_max)
    min_lr_fit = min(max(batch_size / nbs * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    # ------------------------------- 初始化优化器--------------------------------------------
    optimizer = {
        'adam': optim.Adam(model.parameters(), init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        'adamw': optim.AdamW(model.parameters(), init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        'sgd': optim.SGD(model.parameters(), init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    }[optimizer_type]

    # -------------------------------------获取学习率下降公式-------------------------------------
    lr_decay_type = config.get_lr_decay_type()
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, init_lr_fit, min_lr_fit, unfreeze_epoch)

    # -------------------------------------计算每一个世代的长度-------------------------------------
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集!!!")

    # -----------------------------------------加载数据集-----------------------------------------------
    input_shape = config.get_input_size()
    num_classes = config.get_num_classes()
    train_dataset = SegmentationDataset(train_lines, input_shape, num_classes, True, dataset_path)
    val_dataset = SegmentationDataset(val_lines, input_shape, num_classes, False, dataset_path)

    if cuda_enable and cuda_mode == 'ddp':
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, )
        batch_size = batch_size // torch.cuda.device_count()
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    num_workers = config.get_num_workers()
    num_workers = num_workers * torch.cuda.device_count() if cuda_enable else num_workers
    seed = config.get_random_seed()
    gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                     drop_last=True, collate_fn=seg_dataset_collate, sampler=train_sampler,
                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
    gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=seg_dataset_collate, sampler=val_sampler,
                         worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

    weight_save_freq, weight_save_path = config.get_weight_save_param()

    if cuda_enable and cuda_mode == 'ddp':
        dist.barrier()

    for epoch in range(init_epoch, unfreeze_epoch):
        # 当进入解冻阶段，重新设置参数
        if epoch >= freeze_epoch and not unfreeze_flag and freeze_train:
            batch_size = unfreeze_batch_size

            nbs = 16
            lr_limit_max = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
            lr_limit_min = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
            init_lr_fit = min(max(batch_size / nbs * init_lr, lr_limit_min), lr_limit_max)
            min_lr_fit = min(max(batch_size / nbs * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

            lr_scheduler_func = get_lr_scheduler(lr_decay_type, init_lr_fit, min_lr_fit, unfreeze_epoch)

            for param in model.backbone.parameters():
                param.requires_grad = True

            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size

            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

            if cuda_enable and cuda_mode == 'ddp':
                batch_size = batch_size // torch.cuda.device_count()

            gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True, drop_last=True, collate_fn=seg_dataset_collate, sampler=train_sampler,
                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
            gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True, drop_last=True, collate_fn=seg_dataset_collate, sampler=val_sampler,
                                 worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

            unfreeze_flag = True

        if cuda_enable and cuda_mode == 'ddp':
            train_sampler.set_epoch(epoch)

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch(rank=rank, model_train=model_train, model=model, num_classes=num_classes, cur_epoch=epoch,
                      epoch_step=epoch_step, epoch_step_val=epoch_step_val, gen=gen, gen_val=gen_val,
                      total_epoch=unfreeze_epoch, cls_weights=cls_weight, cuda_enable=cuda_enable,
                      optimizer=optimizer, fp16_enable=config.get_fp16(), focal_loss_enable=config.focal_loss_enable(),
                      dice_loss_enable=config.dice_loss_enable(), scaler=scaler, eval_freq=config.eval_freq(),
                      loss_history=loss_history, weight_save_freq=weight_save_freq, weight_save_path=weight_save_path)

        if cuda_enable and cuda_mode == 'ddp':
            dist.barrier()

    if rank == 0:
        loss_history.writer.close()
