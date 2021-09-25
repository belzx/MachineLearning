import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class GoogLeNet(nn.Module):
    def __init__(self, class_num, load_model_path=None, use_gpu=None):
        super(GoogLeNet, self).__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.l2_3 = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.Conv2d(64, 192, 3, padding=1),
            nn.MaxPool2d(3, 2, 1)
        )

        self.l4_7 = nn.Sequential(
            Inception(192, 64, 96, 128, 16, 32, 32),  # 3a
            Inception(256, 128, 128, 192, 32, 96, 64),  # 3b
            nn.MaxPool2d(3, 2, 1),
        )

        self.l8_9 = nn.Sequential(
            Inception(480, 192, 96, 208, 16, 48, 64),  # 4a
        )

        self.l10_15 = nn.Sequential(
            Inception(512, 160, 112, 224, 24, 64, 64),  # 4b
            Inception(512, 128, 128, 256, 24, 64, 64),  # 4c
            Inception(512, 112, 144, 288, 32, 64, 64),  # 4d
        )

        self.l16_17 = nn.Sequential(
            Inception(528, 256, 160, 320, 32, 128, 128),  # 4e
            nn.MaxPool2d(3, 2, 1),
        )

        self.l18_21 = nn.Sequential(
            Inception(832, 256, 160, 320, 32, 128, 128),  # 5a
            Inception(832, 384, 192, 384, 48, 128, 128),  # 5b
            nn.AvgPool2d(7, 1),
            nn.Dropout(p=0.4),  # 40%
        )

        # 5x5 3(V) 不知道如何设置
        self.l8_9_out = nn.Sequential(
            nn.AvgPool2d(5, 3),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1),
        )

        self.l10_15_out = nn.Sequential(
            nn.AvgPool2d(5, 3),
            nn.Conv2d(in_channels=528, out_channels=1024, kernel_size=1, stride=1),
        )
        self.liner_1 = nn.Linear(1024, class_num)

        self.liner_2 = nn.Linear(4 * 4 * 1024, class_num)
        self.liner_3 = nn.Linear(4 * 4 * 1024, class_num)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2_3(x)
        x = self.l4_7(x)
        x = self.l8_9(x)
        out_4a = self.l8_9_out(x)
        x = self.l10_15(x)
        out_4d = self.l10_15_out(x)
        x = self.l16_17(x)
        x = self.l18_21(x)
        x = x.view(-1, 1024)
        out_4a = out_4a.view(-1, 4 * 4 * 1024)
        out_4d = out_4d.view(-1, 4 * 4 * 1024)
        x = self.liner_1(x)
        out_4a = self.liner_2(out_4a)
        out_4d = self.liner_3(out_4d)

        x = F.softmax(x)
        out_4a = F.softmax(out_4a)
        out_4d = F.softmax(out_4d)
        return x, out_4a, out_4d


def conv_relu(in_channel, out_channel, kernel, stride=1, padding=0):
    layer = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel, stride, padding),
        nn.BatchNorm2d(out_channel, eps=1e-3),
        nn.ReLU(True)
    )
    return layer


class Inception(nn.Module):
    def __init__(self, in_channel, out1_1, out2_1, out2_2, out3_1, out3_2, out4_1):
        super(Inception, self).__init__()
        # 第一条线路
        self.branch1x1 = nn.Conv2d(in_channel, out1_1, 1, stride=1, padding=0)

        # 第二条线路
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channel, out2_1, 1, stride=1, padding=1),
            nn.Conv2d(out2_1, out2_2, 3, stride=1),
        )

        # 第三条线路
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channel, out3_1, 1, stride=1, padding=1),
            nn.Conv2d(out3_1, out3_2, 3, stride=1),
        )

        # 第四条线路
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channel, out4_1, 1, stride=1)
        )

    # 定义inception前向传播
    def forward(self, x):
        f1 = self.branch1x1(x)
        f2 = self.branch3x3(x)
        f3 = self.branch5x5(x)
        f4 = self.branch_pool(x)
        output = torch.cat((f1, f2, f3, f4), dim=1)
        return output


if __name__ == '__main__':
    x = torch.randn(10, 3, 224, 224)
    net = GoogLeNet(class_num=1000)
    print(net)
    x1, x2, x3 = net(x)
    print(x1.shape)
    print(x2.shape)
    print(x3.shape)
