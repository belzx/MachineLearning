import os

import torch
import torch.nn as nn
from torch.autograd.grad_mode import F

import utils

"""
YOLO论文地址https://arxiv.org/pdf/1506.02640.pdf
"""


class AlexNet(nn.Module):
    def __init__(self, load_model_path=None, pred_class_num=1000):
        super(AlexNet, self).__init__()
        """
        pred_class_name:最终预测类型数量
        """
        # 论文中条件是2块gpu同时训练，但这里是单块，所以卷积核的数量*2
        # input: 3*224*224
        self.con_1 = nn.Sequential(
            # 第一个卷积层(out:(48 * 2),54,54)
            nn.Conv2d(in_channels=3, out_channels=48 * 2, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            # 第二个卷积层(out:(128 * 2),27,27)
            nn.Conv2d(in_channels=48 * 2, out_channels=128 * 2, kernel_size=5, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 第三个卷积层(out:(192 * 2),13,13)
            nn.Conv2d(in_channels=128 * 2, out_channels=192 * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 第四个卷积层(out:(192 * 2),13,13)
            nn.Conv2d(in_channels=192 * 2, out_channels=192 * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 第五个卷积层(out:(192 * 2),13,13)
            nn.Conv2d(in_channels=192 * 2, out_channels=128 * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 池化：(out:(128 * 2),6,6)
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.liner_2 = nn.Sequential(
            nn.Linear(128 * 6 * 6 * 2, 2048 * 2),
            nn.ReLU(inplace=True),
            nn.Linear(2048 * 2, 2048 * 2),
            nn.ReLU(inplace=True),
            nn.Linear(2048 * 2, pred_class_num),
            # 这里sigmod 与 softmax的区别，sigmod导致突然的梯度消失
            nn.Softmax()
        )

        if load_model_path is not None:
            self.load_state_dict(torch.load(load_model_path))

    def forward(self, x):
        x = self.con_1(x)
        x = x.view(-1, 128 * 6 * 6 * 2)
        x = self.liner_2(x)
        return x


if __name__ == '__main__':
    # 1:模型数量的测试
    root_path = utils.PROJECT_ROOT_PATH
    load_model_path = os.path.join(root_path, r'AlexNet\output\models\alexnet_mnist.pth')
    inputs = torch.randn(1, 3, 224, 224)
    net = AlexNet(pred_class_num=10)
    outputs = net(inputs)
    # print(outputs)
    print(outputs.shape)  # out:torch.Size([64, 7, 7, 30])
    pass
