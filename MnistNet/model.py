import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

"""
"""


class MnistNet(nn.Module):
    def __init__(self,load_model_path, use_gpu):
        super(MnistNet, self).__init__()
        # 1*1*28*28
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)

        if load_model_path is not None:
            if use_gpu:
                a = torch.load(load_model_path, map_location="cuda:0")
            else:
                a = torch.load(load_model_path)
            self.load_state_dict(a)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)  # 1* 10 * 24 *24
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)  # 1* 10 * 12 * 12
        out = self.conv2(out)  # 1* 20 * 10 * 10
        out = F.relu(out)
        out = out.view(in_size, -1)  # 1 * 2000
        out = self.fc1(out)  # 1 * 500
        out = F.relu(out)
        out = self.fc2(out)  # 1 * 10
        # out = F.softmax(out, dim=1)
        return out


if __name__ == '__main__':
    # 1:模型数量的测试
    root_path = utils.PROJECT_ROOT_PATH
    load_model_path = os.path.join(root_path, r'MnistNet\output\models\MnistNet_mnist.pth')
    inputs = torch.randn(1, 1, 28, 28)
    net = MnistNet(load_model_path=load_model_path,use_gpu=True)
    outputs = net(inputs)
    # print(outputs)
    print(outputs.shape)  # out:torch.Size([64, 7, 7, 30])
    pass
