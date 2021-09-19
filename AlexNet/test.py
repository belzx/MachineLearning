"""
"""
import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import utils
from utils import YoloDataset
import model
import torch.nn.functional as F


def mnist_train():
    # mnist数据集一共是 10种类型
    global pred_class_num,use_gpu
    pred_class_num = MNIST_PRED_CLASS_NUM
    if use_gpu:
        if not torch.cuda.is_available():
            use_gpu = False
            print("Not support GPU for train！")

    # 是否加载已有模型
    if load_model_path is not None:
        net = model.AlexNet(load_model_path=load_model_path, pred_class_num=pred_class_num, use_gpu=use_gpu)
    else:
        net = model.AlexNet(pred_class_num=pred_class_num, use_gpu=use_gpu)

    if use_gpu:
        net = net.to(device=torch.device("cuda:0"))

    dataset = utils.MnistDataset(batch_size)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # 转成224*224格式
    t = transforms.Compose([
        transforms.Resize([image_size, image_size])
    ])

    # 总照片数量
    total_num = 0
    # 预测正确
    right = 0
    total_loss = 0.
    for i, (data, target) in enumerate(dataset.train_loader):
        batch_s = data.shape[0]
        total_num += batch_s
        optimizer.zero_grad()
        # 转3*224*224
        data = t(data)
        batch_s = data.shape[0]
        data = data.expand(batch_s, 3, image_size, image_size)  # [batch_size,3，224，224]

        if use_gpu:
            data = data.cuda()
            target = target.cuda()

        pred = net(data)
        loss = F.cross_entropy(pred, target)  # 这里返回的损失值
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        # 正确率
        pred_num, pred_index = pred.max(1)
        for k in range(len(target)):
            if pred_num[k].item() > threshold and pred_index[k].item() == target[k].item():
                right += 1

        print('Iter [%d/%d] Loss: %.4f, average_loss: %.4f, accuracy:  %.4f' % (
             i + 1, len(dataset.train_loader), loss.item(), total_loss / (i + 1),right/total_num))
    # torch.save(net.state_dict(), be_save_model_path)


if __name__ == '__main__':
    root_path = utils.PROJECT_ROOT_PATH
    # MNIST 种类
    MNIST_PRED_CLASS_NUM = 10
    # 训练模型的数据大小
    image_size = 224
    # 学习率可以设置为3、1、0.5、0.1、0.05、0.01、0.005,0.005、0.0001、0.00001
    learning_rate = .1
    # 每次训练的图片数量
    batch_size = 128
    # 保存间隔次数
    per_batch_size_to_save = 30
    # 已有模型
    load_model_path = os.path.join(root_path, r'AlexNet\output\models\alexnet_mnist.pth')
    # 训练好的模型保存路径
    # be_save_model_path = os.path.join(root_path, r'AlexNet\output\models\alexnet_mnist.pth')
    # 是否使用GPU
    use_gpu = True
    # 是预测阈值
    threshold = 0.9

    # 1:测试MNIST
    print("START MNIST TRAIN!")
    pred_class_num = MNIST_PRED_CLASS_NUM
    mnist_train()
    print("END!")

