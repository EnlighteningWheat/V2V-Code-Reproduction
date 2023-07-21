import numpy as np
import torch
import skimage
from skimage.transform import resize
from skimage.io import imread, imsave
from skimage import data, img_as_float
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os
from model import *


# 定义一个名为ScalarDataSet的类
class ScalarDataSet():
    def __init__(self, args):
        # 初始化方法，接收args参数
        self.device = torch.device("cuda:0" if args.cuda else "cpu")  # 设备选择，如果cuda可用，选择cuda，否则选择cpu
        self.dataset = args.dataset  # 获取数据集
        self.crop = args.crop  # 获取是否裁剪的参数
        self.croptimes = args.croptimes  # 获取裁剪次数
        if self.dataset == 'ionization':  # 如果数据集是ionization
            self.dim = [96, 96, 144]  # 定义数据维度
            self.total_samples = 100  # 定义总样本数量
            self.cropsize = [64, 64, 96]  # 定义裁剪尺寸
            self.train_samples = [i for i in range(1, 70)]  # 定义训练样本范围
            self.test_samples = [i for i in range(70, 100)]  # 定义测试样本范围
            self.s = 'ionization_ab_H/train/normInten_'  # 源数据路径
            self.t = 'ionization_ab_H/test/normInten_'  # 目标数据路径

    def ReadData(self):
        # 定义读取数据方法
        self.source = []  # 初始化源数据列表
        self.target = []  # 初始化目标数据列表
        for i in self.train_samples:  # 对于每个训练样本
            print(i)
            s = np.zeros((1, self.dim[0], self.dim[1], self.dim[2]))  # 初始化源数据矩阵
            d = np.fromfile(self.s + '{:04d}'.format(i) + '.raw', dtype='<f')  # 读取训练数据文件

            d = 2 * (d - np.min(d)) / (np.max(d) - np.min(d)) - 1  # 将源数据进行归一化处理
            d = d.reshape(self.dim[2], self.dim[1], self.dim[0]).transpose()  # 将源数据进行形状变换
            s[0] = d  # 将变换后的源数据赋值给s
            self.source.append(s)  # 将t添加到目标数据列表中

        for i in self.test_samples:  # 对于每个测试样本
            print(i)
            t = np.zeros((1, self.dim[0], self.dim[1], self.dim[2]))  # 初始化目标数据矩阵
            o = np.fromfile(self.t + '{:04d}'.format(i) + '.raw', dtype='<f')  # 读取测试数据文件
            o = 2 * (o - np.min(o)) / (np.max(o) - np.min(o)) - 1  # 将目标数据进行归一化处理
            o = o.reshape(self.dim[2], self.dim[1], self.dim[0]).transpose()  # 将目标数据进行形状变换
            t[0] = o  # 将变换后的目标数据赋值给t
            self.target.append(t)  # 将t添加到目标数据列表中

        self.source = np.asarray(self.source)  # 将源数据列表转为numpy数组
        self.target = np.asarray(self.target)  # 将目标数据列表转为numpy数组

    def TrainingData(self):
        # 定义获取训练数据的方法
        if self.crop == 'yes':  # 如果需要裁剪数据
            a = []  # 初始化a列表
            o = []  # 初始化o列表
            for k in range(len(self.source)):  # 对于每个训练样本
                n = 0  # 初始化n为0
                while n < self.croptimes:  # 如果n小于裁剪次数
                    if self.dim[0] == self.cropsize[0]:  # 如果第一维大小等于裁剪大小
                        x = 0  # x赋值为0
                    else:
                        x = np.random.randint(0, self.dim[0] - self.cropsize[0])  # 否则，x为随机数

                    if self.dim[1] == self.cropsize[1]:  # 同理
                        y = 0
                    else:
                        y = np.random.randint(0, self.dim[1] - self.cropsize[1])

                    if self.dim[2] == self.cropsize[2]:
                        z = 0
                    else:
                        z = np.random.randint(0, self.dim[2] - self.cropsize[2])

                    c0 = self.source[k][:, x:x + self.cropsize[0], y:y + self.cropsize[1],
                         z:z + self.cropsize[2]]  # 裁剪源数据
                    o.append(c0)  # 添加到o列表

                    # 扩充目标数据以匹配源数据的数量
                    idx = k % len(self.target)  # 计算目标数据的索引
                    c1 = self.target[idx][:, x:x + self.cropsize[0], y:y + self.cropsize[1],
                         z:z + self.cropsize[2]]  # 裁剪目标数据
                    a.append(c1)  # 添加到a列表

                    n += 1  # n加1

            o = np.asarray(o)  # 将o列表转为numpy数组
            a = np.asarray(a)  # 将a列表转为numpy数组
            a = torch.FloatTensor(a)  # 将a转为FloatTensor
            o = torch.FloatTensor(o)  # 将o转为FloatTensor
        else:
            a = torch.FloatTensor(self.target)  # 将目标数据转为FloatTensor
            o = torch.FloatTensor(self.source)  # 将源数据转为FloatTensor
        dataset = torch.utils.data.TensorDataset(o, a)  # 构建数据集
        train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)  # 创建DataLoader
        return train_loader  # 返回DataLoader