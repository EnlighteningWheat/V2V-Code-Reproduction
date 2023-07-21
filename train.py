# 导入必要的包和模块
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
from model import *


# 训练函数，包括生成器G和判别器D的训练
def train(G, D, ScalarDataset, args):
    # 创建存放损失的文件
    loss_curve = open('ionization_ab_H/Exp/' + 'loss-' + args.dataset + '.txt', 'w')
    # 设置设备为GPU或CPU
    device = torch.device("cuda:0" if args.cuda else "cpu")
    # 为生成器和判别器设置优化器
    optimizer_G = optim.Adam(G.parameters(), lr=args.lr_G, betas=(0.9, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr_D, betas=(0.5, 0.999))
    # 创建MSE损失函数
    L2 = nn.MSELoss()
    critic = 1
    # 迭代训练
    for itera in range(1, 2000 + 1):
        train_loader = ScalarDataset.TrainingData()
        loss_G = 0
        loss_D = 0
        mse_loss = 0
        print("========================")
        print(itera)
        x = time.time()
        for batch_idx, (i, o) in enumerate(train_loader):
            batch = i.size()[0]
            if args.cuda:
                i = i.cuda()
                o = o.cuda()
            for p in G.parameters():
                p.requires_grad = False
            ################################
            # 更新D网络
            ################################
            for j in range(1, critic + 1):
                optimizer_D.zero_grad()
                # 使用真实数据进行训练
                label_real = Variable(torch.full((batch,), 1.0, device=device))
                _, output_real = D(o)
                real_loss = L2(output_real, label_real)

                # 使用生成的假数据进行训练
                fake_data = G(i)
                label_fake = Variable(torch.full((batch,), 0.0, device=device))

                _, output_fake = D(fake_data)
                fake_loss = L2(output_fake, label_fake)

                loss = 0.5 * (real_loss + fake_loss)

                loss.backward()
                loss_D += loss.mean().item()
                optimizer_D.step()

            #################################
            # 更新G网络
            #################################
            for p in G.parameters():
                p.requires_grad = True
            for p in D.parameters():
                p.requires_grad = False
            for j in range(1, 1 + 1):
                optimizer_G.zero_grad()
                label_real = Variable(torch.full((batch,), 1.0, device=device))
                fake_data = G(i)

                features_fake, output_real = D(fake_data)
                features_real, _ = D(o)

                # 对抗性损失
                L_adv = L2(output_real, label_real)

                # 内容损失
                L_c = L2(fake_data, o)

                ### 特征损失
                L_p = L2(features_fake[0], features_real[0]) + L2(features_fake[1], features_real[1]) + L2(
                    features_fake[2], features_real[2]) + L2(features_fake[3], features_real[3])

                # 总损失
                error = 1e-3 * L_adv + 1 * L_c + 1e-2 * L_p
                error.backward()
                loss_G += error.item()
                optimizer_G.step()
            for p in D.parameters():
                p.requires_grad = True
        y = time.time()
        print('Time = ' + str(y - x) + 's')
        # 将每个迭代的损失写入文件
        loss_curve.write("Epochs " + str(itera) + ": Loss = " + str(loss_G))
        loss_curve.write('\n')
        # 在一定的迭代次数保存模型参数
        if itera % 10 == 0 or itera == 1:
            torch.save(G.state_dict(), 'ionization_ab_H/Exp/' + args.dataset + '-' + str(itera) + '-V2V.pth')

    loss_curve.write('\n')
    loss_curve.close()


# 用于进行推理的函数
def inf(args, dataset):
    model = V2V()
    # 加载模型参数
    model.load_state_dict(torch.load('ionization_ab_H/Exp/' + args.dataset + '-' + str(args.epochs) + '-V2V.pth'))
    if args.cuda:
        model.cuda()
    x = time.time()
    for i in range(70, 100):
        print(i)
        v = np.zeros((1, 1, dataset.dim[0], dataset.dim[1], dataset.dim[2]))
        d = np.fromfile(dataset.t + '{:04d}'.format(i) + '.raw', dtype='<f')
        d = 2 * (d - np.min(d)) / (np.max(d) - np.min(d)) - 1
        d = d.reshape(dataset.dim[2], dataset.dim[1], dataset.dim[0]).transpose()
        v[0][0] = d
        v = torch.FloatTensor(v)
        if args.cuda:
            v = v.cuda()
        if args.dataset == 'ionization':
            t = concatsubvolume(model, v, [96, 96, 144], args)
        print(t.min())
        print(t.max())
        t = t.flatten('F')
        t = np.asarray(t, dtype='<f')
        t.tofile('ionization_ab_H/Result/' + args.dataset + '/' + '{:04d}'.format(i) + '.raw', format='<f')
    y = time.time()
    print((y - x) / dataset.total_samples)


# 一个拼接子体积的函数
def concatsubvolume(G, data, win_size, args):
    x, y, z = data.size()[2], data.size()[3], data.size()[4]
    w = np.zeros((win_size[0], win_size[1], win_size[2]))
    for i in range(win_size[0]):
        for j in range(win_size[1]):
            for k in range(win_size[2]):
                dx = min(i, win_size[0] - 1 - i)
                dy = min(j, win_size[1] - 1 - j)
                dz = min(k, win_size[2] - 1 - k)
                d = min(min(dx, dy), dz) + 1
                w[i, j, k] = d
    w = w / np.max(w)
    avI = np.zeros((x, y, z))
    pmap = np.zeros((x, y, z))
    avk = 4
    for i in range((avk * x - win_size[0]) // win_size[0] + 1):
        for j in range((avk * y - win_size[1]) // win_size[1] + 1):
            for k in range((avk * z - win_size[2]) // win_size[2] + 1):
                si = (i * win_size[0] // avk)
                ei = si + win_size[0]
                sj = (j * win_size[1] // avk)
                ej = sj + win_size[1]
                sk = (k * win_size[2] // avk)
                ek = sk + win_size[2]
                if ei > x:
                    ei = x
                    si = ei - win_size[0]
                if ej > y:
                    ej = y
                    sj = ej - win_size[1]
                if ek > z:
                    ek = z
                    sk = ek - win_size[2]
                d = data[:, :, si:ei, sj:ej, sk:ek]
                with torch.no_grad():
                    result = G(d)
                k = np.multiply(result[0][0].cpu().detach().numpy(), w)
                avI[si:ei, sj:ej, sk:ek] += w
                pmap[si:ei, sj:ej, sk:ek] += k
    high = np.divide(pmap, avI)
    return high
