# _*_ coding: utf-8 _*_
# This file is created by C. Zhang for personal use.
# @Time         : 18/08/2022 14:19
# @Author       : tl22089
# @File         : models.py
# @Affiliation  : University of Bristol
import torch
from torchvision import models
from torch import nn
import torch.nn.functional as F
import math


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args
        self.input = nn.Linear(args.ws * 3, 512)
        if args.type != 'metric':
            self.fc = nn.Linear(in_features=512, out_features=args.bs_classes)
        else:
            self.fc = nn.Linear(in_features=512, out_features=args.out_dim)
    def forward(self, x):
        x = F.relu(self.input(x))
        return self.fc(x)


import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.args = args
        # 第一个卷积层：1x7过滤器
        self.conv1 = nn.Conv2d(2, 40, kernel_size=(1, 7))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))
        # 第二个卷积层：1x5过滤器
        self.conv2 = nn.Conv2d(40, 40, kernel_size=(1, 9))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.conv5 = nn.Conv2d(40, 40, kernel_size=(1, 9))
        self.pool5 = nn.MaxPool2d(kernel_size=(1, 2))
        # 第三个卷积层：2x7过滤器
        self.conv3 = nn.Conv2d(40, 40, kernel_size=(2, 7), padding=(1, 3))  # 添加高度和宽度填充
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # 高度减半
        # 第四个卷积层：2x5过滤器
        self.conv4 = nn.Conv2d(40, 40, kernel_size=(2, 9), padding=(1, 2))  # 添加填充
        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(p=0.2)  # p为丢弃概率

        # 全连接层
        self.fc1 = nn.Linear(40320, 1024)  # 根据输入尺寸调整

        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        if args.type != 'metric':
            self.fc4 = nn.Linear(in_features=256, out_features=args.bs_classes)
        else:
            self.fc4 = nn.Linear(in_features=256, out_features=args.out_dim)

    def _convert_data_size(self, x):
        """
        将输入数据拆分为 I、Q 和幅值三个通道，并保留原始格式（不重塑为 32x32）。
        """
        # 根据 args.ws（每个通道的数据长度）拆分数据
        xr = x[:, :self.args.ws]  # I 分量
        xi = x[:, self.args.ws:2 * self.args.ws]  # Q 分量
        # 合并为多通道 (batch_size, 3, ws)
        return torch.stack([xr, xi], dim=1)  # 输出形状 (batch_size, 3, ws)

    def forward(self, x):
        # 输入预处理
        x = self._convert_data_size(x)
        x = x.unsqueeze(2)  # 添加 height 维度，输出形状为 (batch_size, 2, 1, ws)
        #print("输入第一层的大小：", x.shape)
        # 卷积层与池化层
        x = self.pool1(F.relu(self.conv1(x)))
        #print("第一层卷积后的大小：", x.shape)
        x = self.pool2(F.relu(self.conv2(x)))
        #print("第二层卷积后的大小：", x.shape)
        x = self.pool5(F.relu(self.conv5(x)))
        #print("第三层卷积后的大小：", x.shape)
        x = self.pool3(F.relu(self.conv3(x)))
        #print("第四层卷积后的大小：", x.shape)
        x = F.relu(self.conv4(x))
        #print("第5层卷积后的大小：", x.shape)
        # Dropout层
        x = self.dropout(x)
        # 展平数据以输入全连接层
        x = torch.flatten(x, 1)
        #print("flatten层后的大小：", x.shape)
        # 全连接层
        x = F.relu(self.fc1(x))
        #print("fc1层后的大小：", x.shape)
        # Dropout层
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        #print("fc2层后的大小：", x.shape)
        # Dropout层
        x = self.dropout(x)
        features = F.relu(self.fc3(x))
        #print("fc3层后的大小：", features.shape)
        outputs = self.fc4(features)
        #print("输出层后的大小：", outputs.shape)
        # 返回特征和输出
        return features, outputs


class CNN2D(nn.Module):
    def __init__(self, args):
        super(CNN2D, self).__init__()
        self.args = args

        # 第一层卷积，输入通道为 1（单通道），输出通道为 32，卷积核大小为 3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # 池化层，使用最大池化
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(23040, 512)  # 128通道，32*7是池化后的特征图大小
        self.fc2 = nn.Linear(512, args.bs_classes)

    def _convert_data_size(self, x):
        # 假设输入的 x 形状是 (batch_size, 102, 63)
        # 增加一个维度，将其转换为 (batch_size, 1, 102, 63)
        return x.unsqueeze(1)  # 输出形状 (batch_size, 1, 102, 63)

    def forward(self, x):
        # 输入预处理
        #print("输入的大小：", x.shape)
        x = self._convert_data_size(x)
        #x = x.unsqueeze(2)  # 添加 height 维度，输出形状为 (batch_size, 2, 1, ws)
        #print("unsqueeze后的大小：", x.shape)
        # 卷积层 + 激活函数 + 池化
        x = self.pool(F.relu(self.conv1(x)))  # 32通道
        #print("第一层卷积后的大小：", x.shape)
        x = self.pool(F.relu(self.conv2(x)))  # 64通道
        #print("第2层卷积后的大小：", x.shape)
        x = self.pool(F.relu(self.conv3(x)))  # 128通道
        #print("第3层卷积后的大小：", x.shape)
        # 展平
        x = torch.flatten(x, 1)
        #print("展平后的大小：", x.shape)#(64*23040)
        # 全连接层
        features = F.relu(self.fc1(x))
        #print("第一个全连接层后的大小：", features.shape)
        outputs = self.fc2(features)
        #print("输出的大小：", outputs.shape)
        return features, outputs

class RFSignalCNN(nn.Module):
    def __init__(self, args):
        super(RFSignalCNN, self).__init__()
        self.args = args

        # 第一层卷积：16个滤波器，每个滤波器大小为1x4
        self.conv1 = nn.Conv2d(2, 16, kernel_size=(1, 4))
        self.bn1 = nn.BatchNorm2d(16)
        self.leakyrelu1 = nn.LeakyReLU(negative_slope=0.01)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # 第二层卷积：16个滤波器，每个滤波器大小为1x4
        self.conv2 = nn.Conv2d(16, 24, kernel_size=(1, 4))
        self.bn2 = nn.BatchNorm2d(24)
        self.leakyrelu2 = nn.LeakyReLU(negative_slope=0.01)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # 第三层卷积：24个滤波器，每个滤波器大小为1x4
        self.conv3 = nn.Conv2d(24, 32, kernel_size=(1, 4))
        self.bn3 = nn.BatchNorm2d(32)
        self.leakyrelu3 = nn.LeakyReLU(negative_slope=0.01)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # 第四层卷积：32个滤波器，每个滤波器大小为1x4
        self.conv4 = nn.Conv2d(32, 48, kernel_size=(1, 4))
        self.bn4 = nn.BatchNorm2d(48)
        self.leakyrelu4 = nn.LeakyReLU(negative_slope=0.01)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # 第五层卷积：48个滤波器，每个滤波器大小为1x4
        self.conv5 = nn.Conv2d(48, 64, kernel_size=(1, 4))
        self.bn5 = nn.BatchNorm2d(64)
        self.leakyrelu5 = nn.LeakyReLU(negative_slope=0.01)
        self.pool5 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # 第六层卷积：64个滤波器，每个滤波器大小为2x4
        self.conv6 = nn.Conv2d(64, 96, kernel_size=(2, 4), stride=1, padding=(0, 2))
        self.bn6 = nn.BatchNorm2d(96)
        self.leakyrelu6 = nn.LeakyReLU(negative_slope=0.01)
        self.pool6 = nn.AdaptiveAvgPool2d((1, 256))  # 使用自适应平均池化层

        # 全连接层
        self.fc1 = nn.Linear(16 * 1 * 256, 25)  # 16个通道，大小为256的输出
        self.dropout = nn.Dropout(0.5)  # Dropout层，防止过拟合
        self.fc2 = nn.Linear(25, args.bs_classes)  # 输出为25类

    def forward(self, x):
        # 第一层卷积 + 批归一化 + 激活 + 池化
        x = self.pool1(self.leakyrelu1(self.bn1(self.conv1(x))))

        # 第二层卷积 + 批归一化 + 激活 + 池化
        x = self.pool2(self.leakyrelu2(self.bn2(self.conv2(x))))
        # 第三层卷积 + 批归一化 + 激活 + 池化
        x = self.pool3(self.leakyrelu3(self.bn3(self.conv3(x))))
        # 第四层卷积 + 批归一化 + 激活 + 池化
        x = self.pool4(self.leakyrelu4(self.bn4(self.conv4(x))))
        # 第五层卷积 + 批归一化 + 激活 + 池化
        x = self.pool5(self.leakyrelu5(self.bn5(self.conv5(x))))
        # 第六层卷积 + 批归一化 + 激活 + 自适应池化
        x = self.pool6(self.leakyrelu6(self.bn6(self.conv6(x))))
        # 展平操作
        x = torch.flatten(x, 1)
        # 全连接层 + Dropout
        features = self.dropout(F.relu(self.fc1(x)))
        # print("第一个全连接层后的大小：", features.shape)
        # 输出层（分类）
        outputs = self.fc2(x)
        # print("输出的大小：", outputs.shape)
        return features, outputs


class ResNet18(nn.Module):
    def __init__(self, args):
        super(ResNet18, self).__init__()
        self.args = args
        # 第一层卷积，输入通道为 1（单通道），输出通道为 64，卷积核大小为 7
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 定义残差块
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        # 全连接层
        self.fc = nn.Linear(512, self.args.bs_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(Residual18Block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(Residual18Block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _convert_data_size(self, x):
        # 假设输入的 x 形状是 (batch_size, 102, 63)
        # 增加一个维度，将其转换为 (batch_size, 1, 102, 63)
        return x.unsqueeze(1)  # 输出形状 (batch_size, 1, 102, 63)

    def forward(self, x):
        # 输入预处理
        x = self._convert_data_size(x)

        # 第一层卷积 + 批归一化 + 激活函数 + 池化
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # 残差块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 全局平均池化
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        # 全连接层
        features = F.relu(x)  # 提取特征
        outputs = self.fc(features)  # 分类输出

        return features, outputs


class Residual18Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Residual18Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 定义残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 第一层卷积 + 激活
        out = self.bn2(self.conv2(out))  # 第二层卷积
        out += self.shortcut(x)  # 残差连接
        out = F.relu(out)  # 激活函数
        return out


class ResNet34(nn.Module):
    def __init__(self, args):
        super(ResNet34, self).__init__()
        self.args = args

        # 第一层卷积，输入通道为 1（单通道），输出通道为 64，卷积核大小为 7
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 定义残差块
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        # 全连接层
        self.fc = nn.Linear(512, self.args.bs_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(Residual34Block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(Residual34Block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _convert_data_size(self, x):
        # 假设输入的 x 形状是 (batch_size, 102, 63)
        # 增加一个维度，将其转换为 (batch_size, 1, 102, 63)
        return x.unsqueeze(1)  # 输出形状 (batch_size, 1, 102, 63)

    def forward(self, x):
        # 输入预处理
        x = self._convert_data_size(x)

        # 第一层卷积 + 批归一化 + 激活函数 + 池化
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # 残差块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 全局平均池化
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        # 全连接层
        features = F.relu(x)  # 提取特征
        outputs = self.fc(features)  # 分类输出

        return features, outputs


class Residual34Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Residual34Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 定义残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 第一层卷积 + 激活
        out = self.bn2(self.conv2(out))  # 第二层卷积
        out += self.shortcut(x)  # 残差连接
        out = F.relu(out)  # 激活函数
        return out






