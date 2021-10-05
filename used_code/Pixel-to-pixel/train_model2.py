import argparse
import os
import time
from distutils.util import strtobool
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from Dataset import qmapDataset
from MSELoss import MSELoss, MAELoss
from pytorch_ssim import SSIM

ssim_loss = SSIM()


"""

containing traing and validation code, fit for CPU GPU Multi-GPU training,
results (losses) are saved and written to tensorboard

use new package for densenet and resnet
try: https://pypi.org/project/segmentation-models-pytorch/0.0.3/

"""

def parse_config():
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument("--use_cuda", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--seed", type=int, default=2020)

    # CNN architecture
    parser.add_argument("--arch", type=str, default="fcns", help='options: FCN32s, fcn_resnet101')
    parser.add_argument("--modelname", type=str, default="test_fcns_fixdatasetbug", help='name of model')

    # training dataset
    parser.add_argument("--trainset", type=str, default="/home/x227guo/workspace/SYDE675/Pixel_to_pixel/dataset/train/")
    parser.add_argument("--testset", type=str, default="/home/x227guo/workspace/SYDE675/Pixel_to_pixel/dataset/valid/")
    parser.add_argument("--patch_num", type=int, default=1)
    parser.add_argument("--crop_size", type=int, default=256)

    # hyperparameters
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--initial_lr", type=float, default=1e-3)
    parser.add_argument("--decay_interval", type=int, default=100)
    parser.add_argument("--decay_ratio", type=float, default=0.5)
    parser.add_argument("--resume_epoch", type=int, default=0)

    # utils
    parser.add_argument("--num_workers", type=int, default=12, help='num of threads to load data')
    parser.add_argument("--epochs_per_save", type=int, default=10)
    parser.add_argument('--ckpt_path', default='./checkpoint', type=str, metavar='PATH', help='path to checkpoints')
    parser.add_argument('--board', default='./board', type=str, help='tensorboard log file path')

    return parser.parse_args()


class Trainer(object):

    def __init__(self, config):
        torch.manual_seed(config.seed)
        self.use_cuda = torch.cuda.is_available() and config.use_cuda

        # dataset
        self.train_transform = transforms.Compose([transforms.ToTensor()])
        #,  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        self.train_batch_size = config.batch_size
        self.train_data = qmapDataset(dataroot=config.trainset, 
                                        transforms=self.train_transform, 
                                        patch_num=config.patch_num, 
                                        crop_size=config.crop_size)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=self.train_batch_size,
                                       shuffle=True,
                                       pin_memory=True,
                                       num_workers=config.num_workers)
        self.train_data_size = len(self.train_loader.dataset)
        self.num_steps_per_epoch = len(self.train_loader)

        print('datasize: ', self.train_data_size)
        print('step per epoch: ', self.num_steps_per_epoch)

        # test dataset
        self.test_data = qmapDataset(dataroot=config.testset, 
                                        transforms=self.train_transform, 
                                        patch_num=config.patch_num, 
                                        crop_size=config.crop_size)
        self.test_loader = DataLoader(self.test_data,
                                       batch_size=self.train_batch_size,
                                       shuffle=False,
                                       pin_memory=True,
                                       num_workers=config.num_workers)

        print('datasize: ', len(self.test_loader.dataset))
        print('step per epoch: ', len(self.test_loader))

        # initialize the model
        if config.arch.lower() == "fcn_vgg":
            from fcn import FCN_VGG
            from fcn import VGGNet
            vgg_model = VGGNet(pretrained=True, requires_grad=True, remove_fc=True)
            self.model = FCN_VGG(pretrained_net=vgg_model, n_class=1)
        elif config.arch.lower() == "fcns":
            from fcn import FCNs
            self.model = FCNs(n_class=1)
        elif config.arch.lower() == "fcn_resnet50":
            from fcn import fcn_resnet50_p2p
            self.model = fcn_resnet50_p2p(pretrained=False)
        elif config.arch.lower() == "fcn_resnet101":
            from fcn import fcn_resnet101_p2p
            self.model = fcn_resnet101_p2p(pretrained=False)
        elif config.arch.lower() in ["densenet121", "densenet"]:
            import segmentation_models_pytorch as smp
            # self-made densenet ?
            self.model = smp.Unet('densenet121', classes=1)
        elif config.arch.lower() in ["resnet18", "resnet"]:
            import segmentation_models_pytorch as smp
            self.model = smp.Unet('resnet18', classes=1)
        elif config.arch.lower() == "resnet50":
            import segmentation_models_pytorch as smp
            self.model = smp.Unet('resnet50', classes=1)
            #self.model = torchvision.models.resnet50(num_classes=1)
        elif config.arch.lower() == "u-net":
            from fcn import U_net
            self.model = U_net()
        elif config.arch.lower() == "u-net32":
            from fcn import U_net32
            self.model = U_net32()
        else:
            raise NotImplementedError(f"[****] '{config.arch}' is not a valid architecture")
        self.model_name = type(self.model).__name__
        num_param = sum([p.numel() for p in self.model.parameters()])
        print(f"[*] Initilizing model: {self.model_name}, num of params: {num_param}")

        if torch.cuda.device_count() > 1 and config.use_cuda:
            print("[*] GPU #", torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)

        if self.use_cuda:
            self.model.cuda()

        # loss function
        self.crit_mse = MSELoss()
        self.crit_mae = MAELoss()
        self.crit_cross_entropy = nn.BCELoss()
        if self.use_cuda:
            self.crit_mse = self.crit_mse.cuda()
            self.crit_mae = self.crit_mae.cuda()
            self.crit_cross_entropy = self.crit_cross_entropy.cuda()

        # statistic
        self.start_epoch = 0
        self.train_loss = []
        self.test_results = []

        # optimizer
        self.initial_lr = config.initial_lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.initial_lr)

        # lr scheduler
        self.scheduler = lr_scheduler.StepLR(self.optimizer, 
                                                last_epoch=self.start_epoch-1, 
                                                step_size=config.decay_interval, 
                                                gamma=config.decay_ratio)

        self.max_epochs = config.max_epochs
        self.epochs_per_save = config.epochs_per_save

        self.ckpt_path = os.path.join(config.ckpt_path, config.modelname)
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)

        # resume
        self.pre_epoch = 0
        if not config.resume_epoch == 0:
            model_name = '{}-{:0>5d}.pt'.format(self.model_name, config.resume_epoch)
            model_name = os.path.join(self.ckpt_path, model_name)
            self._load_checkpoint(model_name)
            self.pre_epoch = config.resume_epoch + 1

        self.board_path = os.path.join(config.board, config.modelname)
        self.writer = SummaryWriter(log_dir=self.board_path)

        self.arch = config.arch

    def fit(self):
        for epoch in range(self.max_epochs):
            self._train_single_epoch(epoch)

    def _train_single_epoch(self, epoch):
        self.current_epoch = epoch + self.start_epoch
        epoch = self.current_epoch
        local_counter = epoch * self.num_steps_per_epoch + 1
        start_time = time.time()
        beta = 0.9
        running_loss = 0 if epoch == 0 else self.train_loss[-1]
        loss_corrected = 0.0

        loss_list = []

        # start training
        for step, sample_batched in enumerate(self.train_loader, 0):
            images_batch, grad_batch = sample_batched['dis'], sample_batched['qmap']

            image = Variable(images_batch)  # shape: (batch_size, channel, H, W)
            grad = Variable(grad_batch)  # shape: (batch_size, channel, H, W)

            if self.use_cuda:
                grad = grad.cuda()
                image = image.cuda()

            self.optimizer.zero_grad()
            q = self.model(image)

            self.loss = self.crit_mae(q, grad)
            loss_list.append(self.loss.item() * image.shape[0])

            # statistics
            loss = self.loss.data.item()
            running_loss = beta * running_loss + (1 - beta) * loss
            loss_corrected = running_loss / (1 - beta ** local_counter)

            self.loss.backward()
            self.optimizer.step()

            local_counter += 1
            # start_time = time.time()

        current_time = time.time()
        duration = current_time - start_time
        examples_per_sec = self.num_steps_per_epoch / duration

        lr = self.optimizer.param_groups[0]['lr']

        format_str = '(E:%d) [loss = %.4f, lr = %.6e] (%.1f steps/sec; %.3f sec/epoch)'
        print_str = format_str % (epoch, loss_corrected, lr, examples_per_sec, duration)
        print(print_str)

        mae_loss_aver = np.sum(loss_list) / self.train_data_size

        self.writer.add_scalar('Train/TrainLoss', loss_corrected, epoch)
        self.writer.add_scalar('Train/TrainLoss_mae', mae_loss_aver, epoch)
        self.writer.add_scalar('lr', lr, epoch)

        self.train_loss.append(loss_corrected)
        self.scheduler.step()

        if (epoch + 1) % self.epochs_per_save == 0:
            model_name = '{}-{:0>5d}.pt'.format(self.model_name, epoch)
            model_name = os.path.join(self.ckpt_path, model_name)
            self._save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'train_loss': self.train_loss,
                'test_results': self.test_results,
            }, model_name)

            self._test_single_epoch(epoch)

    # save checkpoint
    @staticmethod
    def _save_checkpoint(state, filename):
        torch.save(state, filename)

    def _test_single_epoch(self, epoch):
        start_time = time.time()

        loss_list = []

        # start training
        for step, sample_batched in enumerate(self.test_loader, 0):
            images_batch, grad_batch = sample_batched['dis'], sample_batched['qmap']

            image = Variable(images_batch)  # shape: (batch_size, channel, H, W)
            grad = Variable(grad_batch)  # shape: (batch_size, channel, H, W)

            if self.use_cuda:
                grad = grad.cuda()
                image = image.cuda()

            self.model.eval()
            q = self.model(image)
            self.model.train()

            self.loss = self.crit_mae(q, grad)
            loss_list.append(self.loss.item())

        current_time = time.time()
        duration = current_time - start_time
        examples_per_sec = self.num_steps_per_epoch / duration

        loss_mean = sum(loss_list) / len(loss_list)

        format_str = 'Testing: (E:%d) [loss = %.4f] (%.1f steps/sec; %.3f sec/epoch)'
        print_str = format_str % (epoch, loss_mean, examples_per_sec, duration)
        print(print_str)

        self.writer.add_scalar('Test/TestLoss', loss_mean, epoch)

    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            print("[*] loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt)
            self.start_epoch = checkpoint['epoch']+1
            self.train_loss = checkpoint['train_loss']
            self.test_results = checkpoint['test_results']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.initial_lr is not None:
                print('?')
                for param_group in self.optimizer.param_groups:
                    param_group['initial_lr'] = self.initial_lr*(0.5**float(self.start_epoch//100))
                    param_group['lr'] = self.initial_lr*(0.5**float(self.start_epoch//100))
            print("[*] loaded checkpoint '{}' (epoch {})"
                  .format(ckpt, checkpoint['epoch']))
        else:
            print("[!] no checkpoint found at '{}'".format(ckpt))


if __name__ == "__main__":
    cfg = parse_config()
    t = Trainer(cfg)
    t.fit()
