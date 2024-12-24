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
        self.f_scores = []
        self.val_loss = []
        self.val_f_scores = []

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

    def append_loss(self, epoch, loss, f_score):
        """
        记录loss同时记录f_score
        :param epoch:
        :param loss:
        :param f_score:
        :return:
        """
        self.losses.append(loss)
        self.f_scores.append(f_score)

        with open(os.path.join(self.log_dir, 'loss.txt'), 'a', encoding='utf-8') as f:
            f.write(str(loss))
            f.write('\n')
        with open(os.path.join(self.log_dir, 'f_score.txt'), 'a', encoding='utf-8') as f:
            f.write(str(f_score))
            f.write('\n')

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('f_score', f_score, epoch)
        self.loss_plot()

    def append_val_loss(self, epoch, val_loss, val_f_score):
        """
        添加val_loss
        :param val_f_score:
        :param epoch:
        :param val_loss:
        :return:
        """
        self.val_loss.append(val_loss)
        self.val_f_scores.append(val_f_score)

        with open(os.path.join(self.log_dir, 'val_loss.txt'), 'a', encoding='utf-8') as f:
            f.write(str(val_loss))
            f.write('\n')

        with open(os.path.join(self.log_dir, 'val_f_score.txt'), 'a', encoding='utf-8') as f:
            f.write(str(val_f_score))
            f.write('\n')

        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.writer.add_scalar('val_f_score', val_f_score, epoch)
        self.val_loss_plot()

    def val_loss_plot(self):
        """
        绘制val_loss图像
        :return:
        """
        iters = range(len(self.val_loss))

        plt.figure()
        plt.plot(iters, self.val_loss, 'red', linewidth=2, label='val loss')
        plt.plot(iters, self.val_f_scores, 'blue', linewidth=2, label='val f-score')

        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth val loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_f_scores, num, 3), 'yellow', linestyle='--', linewidth=2,
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
        """
        绘制loss和f_score图像
        :return:
        """
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.f_scores, 'blue', linewidth=2, label='f-score')

        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.f_scores, num, 3), 'yellow', linestyle='--', linewidth=2,
                     label='smooth f-score')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.cla()
        plt.close('all')
