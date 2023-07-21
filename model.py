# 导入所需的库
import torch.nn as nn
import torch
import torch.autograd as autograd
from torch.nn import init
from torch.nn.modules import conv, Linear
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
import time
import argparse
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn.parallel
import torch.nn.utils.spectral_norm as spectral_norm
import numbers
import math
import os
from torch import Tensor
from torch.nn import Parameter
import yaml


# 网络函数

def weights_init_kaiming(m):
    # 权重初始化函数，使用Kaiming方法初始化权重
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        # 如果是卷积层，使用kaiming_uniform_方法初始化权重
        init.kaiming_uniform_(m.weight.data)
    elif classname.find("Linear") != -1:
        # 如果是线性层，使用kaiming_uniform_方法初始化权重
        init.kaiming_uniform_(m.weight.data)
    elif classname.find("BatchNorm") != -1:
        # 如果是批量归一化层，使用正态分布初始化权重，均值为1，标准差为0.02，偏置设置为0
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class Block(nn.Module):
    # 定义一个基本的卷积块，包含卷积层，ReLU激活函数，以及可选的Dropout层
    def __init__(self, inchannels, outchannels, dropout, kernel, bias, depth, mode, factor=2):
        super(Block, self).__init__()
        layers = []
        for i in range(int(depth)):
            layers += [
                nn.Conv3d(inchannels, inchannels, kernel_size=kernel, padding=kernel // 2, bias=bias),
                nn.ReLU(inplace=True)]
            if dropout:
                layers += [nn.Dropout(0.5)]
        self.model = nn.Sequential(*layers)
        if factor == 2:
            self.stride = 2
            self.padding = 1
        elif factor == 4:
            self.stride = 1
            self.padding = 0
        if mode == 'down':
            self.conv1 = nn.Conv3d(inchannels, outchannels, 4, stride=self.stride, padding=self.padding, bias=bias)
            self.conv2 = nn.Conv3d(inchannels, outchannels, 4, stride=self.stride, padding=self.padding, bias=bias)
        elif mode == 'up':
            self.conv1 = nn.ConvTranspose3d(inchannels, outchannels, 4, stride=self.stride, padding=self.padding,
                                            bias=bias)
            self.conv2 = nn.ConvTranspose3d(inchannels, outchannels, 4, stride=self.stride, padding=self.padding,
                                            bias=bias)
        elif mode == 'same':
            self.conv1 = nn.Sequential(*[
                nn.Conv3d(inchannels, outchannels, kernel_size=kernel, padding=kernel // 2, stride=1, bias=bias)
            ])

            self.conv2 = nn.Sequential(*[
                nn.Conv3d(inchannels, outchannels, kernel_size=kernel, padding=kernel // 2, stride=1, bias=bias)
            ])

    def forward(self, x):
        y = self.model(x)
        y = self.conv1(y)
        x = self.conv2(x)
        return x + y


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.kaiming_uniform_(m.weight.data)
    elif classname.find("Linear") != -1:
        init.kaiming_uniform_(m.weight.data)
    elif classname.find("BatchNorm") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class V2V(nn.Module):
    # V2V模型的定义
    def __init__(self, init_channels=16):
        super(V2V, self).__init__()

        self.conv1 = nn.Conv3d(1, init_channels, 4, 2, 1)
        self.conv2 = nn.Conv3d(init_channels, 2 * init_channels, 4, 2, 1)
        self.conv3 = nn.Conv3d(2 * init_channels, 4 * init_channels, 4, 2, 1)
        self.conv4 = nn.Conv3d(4 * init_channels, 8 * init_channels, 4, 2, 1)

        ### upsample
        self.deconv4 = nn.ConvTranspose3d(8 * init_channels, 4 * init_channels, 4, 2, 1)
        self.conv_u4 = nn.Conv3d(8 * init_channels, 4 * init_channels, 3, 1, 1)
        self.deconv3 = nn.ConvTranspose3d(4 * init_channels, 2 * init_channels, 4, 2, 1)
        self.conv_u3 = nn.Conv3d(4 * init_channels, 2 * init_channels, 3, 1, 1)
        self.deconv2 = nn.ConvTranspose3d(2 * init_channels, init_channels, 4, 2, 1)
        self.conv_u2 = nn.Conv3d(2 * init_channels, init_channels, 3, 1, 1)
        self.deconv1 = nn.ConvTranspose3d(init_channels, init_channels // 2, 4, 2, 1)
        self.conv_u1 = nn.Conv3d(init_channels // 2, 1, 3, 1, 1)

        self.b3 = Block(inchannels=4 * init_channels, outchannels=4 * init_channels, dropout=False, kernel=3,
                        bias=False, depth=2, mode='same', factor=2)
        self.b2 = Block(inchannels=2 * init_channels, outchannels=2 * init_channels, dropout=False, kernel=3,
                        bias=False, depth=2, mode='same', factor=2)
        self.b1 = Block(inchannels=init_channels, outchannels=init_channels, dropout=False, kernel=3, bias=False,
                        depth=2, mode='same', factor=2)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))

        ## upsample

        u1 = F.relu(self.deconv4(x4))
        u1 = F.relu(self.conv_u4(torch.cat((self.b3(x3), u1), dim=1)))
        u2 = F.relu(self.deconv3(u1))
        u2 = F.relu(self.conv_u3(torch.cat((self.b2(x2), u2), dim=1)))
        u3 = F.relu(self.deconv2(u2))
        u3 = F.relu(self.conv_u2(torch.cat((self.b1(x1), u3), dim=1)))
        u4 = F.relu(self.deconv1(u3))
        out = self.conv_u1(u4)
        out = torch.tanh(out)
        return out


class Dis(nn.Module):
    # Dis模型的定义，主要用于鉴别器部分
    def __init__(self):
        super(Dis, self).__init__()
        ### downsample
        self.conv1 = spectral_norm(nn.Conv3d(1, 32, 4, 2, 1), eps=1e-4)
        self.conv2 = spectral_norm(nn.Conv3d(32, 64, 4, 2, 1), eps=1e-4)
        self.conv3 = spectral_norm(nn.Conv3d(64, 128, 4, 2, 1), eps=1e-4)
        self.conv4 = spectral_norm(nn.Conv3d(128, 1, 4, 2, 1), eps=1e-4)
        self.ac = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.ac(self.conv1(x))
        x2 = self.ac(self.conv2(x1))
        x3 = self.ac(self.conv3(x2))
        x4 = self.ac(self.conv4(x3))
        x5 = F.avg_pool3d(x4, x4.size()[2:])
        return [x1, x2, x3, x4], x5.view(-1)
