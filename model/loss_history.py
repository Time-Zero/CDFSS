import matplotlib
import scipy.signal

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
import torch
from colorama import Fore, Style


class LossHistory:
    def __init__(self, log_dir, model, input_shape):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.writer = SummaryWriter(log_dir=self.log_dir)
        # ---------------------- 尝试绘制模型结构------------------
        try:
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            print(Fore.RED + "LossHistory模块生成模型结构失败" + Style.RESET_ALL)
            pass

    def append_loss(self, epoch, loss):
        self.losses.append(loss)
        with open(os.path.join(self.log_dir, 'loss.txt'), 'w', encoding='utf-8') as f:
            f.write(str(loss))
            f.write('\n')

        self.writer.add_scalar('loss', loss, epoch)

    def append_val_loss(self, epoch, val_loss):
        self.val_loss.append(val_loss)
        with open(os.path.join(self.log_dir, 'val_loss.txt'), 'w', encoding='utf-8') as f:
            f.write(str(val_loss))
            f.write('\n')

        self.writer.add_scalar('val_loss', val_loss, epoch)

    def val_loss_plot(self):
        iters = range(len(self.val_loss))

        plt.figure()
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')

        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Val_Loss')
        plt.legend(loc='upper right')

        plt.savefig(os.path.join(self.log_dir, 'epoch_val_loss.png'))

        plt.cla()
        plt.close('all')

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')

        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.cla()
        plt.close('all')
