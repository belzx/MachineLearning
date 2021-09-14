import torch
import torch.nn as nn

"""
YOLO论文地址https://arxiv.org/pdf/1506.02640.pdf
"""


class YoloV1Net(nn.Module):
    def __init__(self, load_model_path=None):
        super(YoloV1Net, self).__init__()
        con_1 = nn.Sequential(
            nn.Conv2d(3, out_channels=64, kernel_size=7, stride=2),  # out:[batch_size, 64, 221, 221]
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # input:112*112*64
        con_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # input:56*56*192
        con_3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),  # 这里padding必须为1 不太懂
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # input:28*28*512
        con_4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),

            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),

            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),

            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # input:14*14*1024
        con_5 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),

            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),

            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),  # 加快训练 论文中没有
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
        )
        # input:7*7*1024
        con_6 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),  # output:7*7*1024
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
        )

        self.net = nn.Sequential(con_1,
                                 con_2,
                                 con_3,
                                 con_4,
                                 con_5,
                                 con_6,
                                 )

        # 全连接层
        self.liner_1 = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 7 * 7 * 30),  # out:batch_size, 7 * 7 * 30
            nn.Sigmoid()  # sigmod处理，映射到0~1
        )

        if load_model_path is not None:
            self.load_state_dict(torch.load(load_model_path))

    def forward(self, x):
        # x:3x448*448
        x = self.net(x)
        x = x.view(-1, 7 * 7 * 1024)
        x = self.liner_1(x)
        # 转为batch_size, 7, 7, 30
        x = x.view(-1, 7, 7, 30)
        return x

if __name__ == '__main__':
    # 1:模型数量的测试
    inputs = torch.randn(1, 3, 448, 448)
    net = YoloV1Net()
    outputs = net(inputs)
    print(outputs)
    print(outputs.shape)  # out:torch.Size([64, 7, 7, 30])
    print("END")
