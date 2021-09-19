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


def mnist_predict(img_pah):
    # mnist数据集一共是 10种类型
    global pred_class_num, use_gpu
    pred_class_num = MNIST_PRED_CLASS_NUM
    if use_gpu:
        if not torch.cuda.is_available():
            use_gpu = False
            print("Not support GPU for train！")

    # 是否加载已有模型
    if load_model_path is not None:
        net = model.AlexNet(load_model_path=load_model_path, pred_class_num=pred_class_num)
    else:
        raise
    #
    image = cv2.imread(img_pah)
    # 设为单通道灰色
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 调整大小
    img = cv2.resize(image, (image_size, image_size))

    # 转格式
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 转tensor调整为(3*224*224)
    img = transform(img)
    img = img.expand(3, image_size, image_size)
    img = img.unsqueeze(0)

    ii = np.transpose(img[0], (1, 2, 0))
    plt.imshow(ii)
    plt.show()

    pred = net(img)


    pred = F.softmax(pred, dim=1)
    _, index = pred.max(1)
    if _[0].item() > 0.2:
        print("Result is : [%s]" % (str(index[0]),))
    else:
        print("No results")



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
    predict_image_path = None
    # 1:测试MNIST
    print("START MNIST TRAIN!")
    pred_class_num = MNIST_PRED_CLASS_NUM
    mnist_predict(os.path.join(root_path, r'AlexNet\img.png'))
    print("END!")
