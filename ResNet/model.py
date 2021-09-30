import torch
from torch import nn, Tensor
from torchvision.models import ResNet
import torch.nn.functional as F


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.con_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(3, 2, 1),
        )

        self.con_2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(3, 2, 1),
        )

        self.k1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )

        self.k2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )

        self.k3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        )

        self.k4 = nn.Conv2d(64, 128, 1, 2, 0)

        self.k5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        )

        self.k6 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        )

        self.k7 = nn.Conv2d(128, 256, 1, 2, 0)

        self.k8 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        )

        self.k9 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        )

        self.k10 = nn.Conv2d(256, 512, 1, 2, 0)

        self.k11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        )

    def _resnet_v1(self, x: Tensor) -> Tensor:
        x = self.k1(x) + x

        x = self.k2(x) + x

        return x

    def _resnet_v2(self, x: Tensor) -> Tensor:
        x1 = self.k4(x)  # 升维度,降低分辨率
        x = self.k3(x) + x1
        x = self.k5(x) + x
        return x

    def _resnet_v3(self, x: Tensor) -> Tensor:
        x1 = self.k7(x)  # 升维度,降低分辨率
        x = self.k6(x) + x1
        x = self.k8(x) + x
        return x

    def _resnet_v4(self, x: Tensor) -> Tensor:
        x1 = self.k10(x)  # 升维度,降低分辨率
        x = self.k9(x) + x1
        x = self.k11(x) + x
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.con_1(x)
        x = self._resnet_v1(x)
        x = self._resnet_v2(x)
        x = self._resnet_v3(x)
        x = self._resnet_v4(x)
        x = nn.AvgPool2d(7)(x)
        x = x.view(-1, 512)
        x = nn.Linear(512, 1000)(x)
        x = F.softmax(x, dim=1)
        return x


if __name__ == '__main__':
    a = torch.zeros(64, 3, 224, 224)
    r = ResNet18()
    x = r(a)
    print(x.shape)
