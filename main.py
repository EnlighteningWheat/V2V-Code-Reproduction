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
from dataio import *
from model import *
from train import *

# 解析命令行参数
parser = argparse.ArgumentParser(description='PyTorch Implementation of V2V')
parser.add_argument('--lr_G', type=float, default=1e-4, metavar='LR',
                    help='学习率 of G')
parser.add_argument('--lr_D', type=float, default=4e-4, metavar='LR',
                    help='学习率 of D')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='禁用CUDA训练')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='训练的输入批量大小')
parser.add_argument('--dataset', type=str, default = 'ionization',
                    help='数据集')
parser.add_argument('--mode', type=str, default='train' ,
                    help='训练或推理')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='训练的轮数 (默认: 500)')
parser.add_argument('--croptimes', type=int, default=4, metavar='N',
                    help='每个数据的裁剪次数')
parser.add_argument('--crop', type=str, default='yes', metavar='N',
                    help='是否裁剪数据')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 30, 'pin_memory': True} if args.cuda else {}
args.mode = 'inf'

def main():
    # 判断模式，训练或推理
    if args.mode == 'train':
        # 读取数据集
        DataSet = ScalarDataSet(args)
        DataSet.ReadData()

        # 设置随机种子
        torch.manual_seed(np.random.randint(int(2 ** 31) - 1))
        np.random.seed(np.random.randint(int(2 ** 31) - 1))

        # 初始化模型
        Model = V2V()
        D = Dis()

        # 如果cuda可用，使用cuda
        if args.cuda:
            Model.cuda()
            D.cuda()
        # 使用kaiming方法初始化模型参数
        Model.apply(weights_init_kaiming)

        D.apply(weights_init_kaiming)

        # 训练模型
        train(Model,D,DataSet,args)

    elif args.mode == 'inf':
        # 读取数据集
        DataSet = ScalarDataSet(args)
        # 推理
        inf(args,DataSet)

# 主函数入口
if __name__== "__main__":
    main()
