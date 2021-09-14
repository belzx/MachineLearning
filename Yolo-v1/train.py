"""
模型训练的内容为VOC2007
"""
import os

import torch
from torch.utils.data import DataLoader

import utils
from utils import MyDataset
import model
import yolo_loss


def train():
    global use_gpu
    if use_gpu:
        if not torch.cuda.is_available():
            use_gpu = False
            print("Not support GPU for train！")
    # 加载训练信息
    train_dataset = MyDataset(xml_dir, image_dir, classname_to_index_file_path, image_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # 是否加载已有模型
    if load_model_path is not None:
        net = model.YoloV1Net(load_model_path=load_model_path)
    else:
        net = model.YoloV1Net()
    net.train()
    #
    criterion = yolo_loss.YoloLoss(7, 2, 5, 0.5)
    for k in range(num_epochs):
        # 优化器的使用策略
        if k == 0:
            optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        elif k == 5:
            optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate / 10, momentum=0.9, weight_decay=5e-4)
        elif k == 100:
            optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate / 100, momentum=0.9, weight_decay=5e-4)
        total_loss = 0.
        for i, (images, target) in enumerate(train_loader):
            pred = net(images)
            # 获取损失
            loss = criterion(pred, target)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('epoch [%d]/[%d] Iter [%d/%d] Loss: %.4f, average_loss: %.4f' % (
                k + 1, num_epochs, i + 1, len(train_loader), loss.item(), total_loss / (i + 1)))
            if i > 0 and i % per_batch_size_to_save == 0:
                torch.save(net.state_dict(), be_save_model_path)
        torch.save(net.state_dict(), be_save_model_path)


if __name__ == '__main__':
    root_path = utils.PROJECT_ROOT_PATH
    xml_dir = os.path.join(root_path, r'data\train\VOC2007\Annotations')
    image_dir = os.path.join(root_path, r'data\train\VOC2007\JPEGImages')
    classname_to_index_file_path = os.path.join(root_path, r'data\voc2007-class-to-index.txt')
    image_size = 448
    # 学习率可以设置为3、1、0.5、0.1、0.05、0.01、0.005,0.005、0.0001、0.00001
    learning_rate = 0.001
    # 数据集训练次数
    num_epochs = 5
    # 每次训练的图片数量
    batch_size = 10
    # 保存间隔次数
    per_batch_size_to_save = 5
    # 已有模型
    # load_model_path = os.path.join(root_path, r'Yolo-v1\output\models\best.pth')
    load_model_path = None
    # 训练好的模型保存路径
    be_save_model_path = os.path.join(root_path, r'Yolo-v1\output\models\best.pth')
    # 是否使用GPU
    use_gpu = True
    print("START!")
    train()
    print("END!")
