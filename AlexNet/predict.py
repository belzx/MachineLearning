"""
"""
import os

import cv2
import torch
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
    image = cv2.imread(predict_image_path)

    # 调整大小
    img = cv2.resize(image, (image_size, image_size))
    transform = transforms.Compose([transforms.ToTensor(), ])

    # 转tensor调整为(3*224*224)
    img = transform(img)
    img = img.unsqueeze(0)

    pred = net(img)

    _, index = pred.max(1)
    if _[0].item() > threshold:
        print("Result is : [%s]" % (str(index[0]),))
    else:
        print("No results")


if __name__ == '__main__':
    root_path = utils.PROJECT_ROOT_PATH
    # MNIST 种类
    MNIST_PRED_CLASS_NUM = 10
    # 训练模型的数据大小
    image_size = 224
    # 每次测试的图片数量
    batch_size = 64
    # 保存间隔次数
    per_batch_size_to_save = 5
    # 已有模型
    load_model_path = os.path.join(root_path, r'AlexNet\output\models\alexnet_mnist.pth')
    # 是否使用GPU
    use_gpu = True
    # 训练好的模型保存路径
    predict_image_path = os.path.join(root_path, r'AlexNet\xx.png')
    # be_save_model_path = None
    # 阈值
    threshold = 0.9
    # 1:测试MNIST
    print("START MNIST TRAIN!")
    pred_class_num = MNIST_PRED_CLASS_NUM
    mnist_predict()
    print("END!")

    # 2:训练TODO
