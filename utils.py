import math
import os
import xml
import xml.dom.minidom
import cv2
from torch import tensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import torch
from PIL import Image
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

# 当前项目跟路径
from torchvision import datasets

PROJECT_ROOT_PATH = os.path.abspath(os.path.dirname(__file__))


class MnistDataset:
    def __init__(self, batch_size=64):
        train_dataset = datasets.MNIST(os.path.join(PROJECT_ROOT_PATH, "data"),
                                       train=True,
                                       transform=transforms.Compose([transforms.ToTensor(),
                                                                     # transforms.Normalize((0.1307,), (0.3081,)),
                                                                     ]),
                                       download=True,

                                       )
        test_dataset = datasets.MNIST(os.path.join(PROJECT_ROOT_PATH, "data"), train=False,
                                      transform=transforms.Compose([transforms.ToTensor(),
                                                                    # transforms.Normalize((0.1307,), (0.3081,)),
                                                                    ]),
                                      download=True)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=False)


class YoloDataset(Dataset):
    def __init__(self, xml_dir, image_dir, dict_class_index_file, transform_size,
                 grid_num=7):
        """
        :param xml_dir: xml文件地址
        :param image_dir: 图片地址
        """

        """
        data_set结构如下
        [
            [
              filename, # 文件名称
              file_height, # 文件大小
              file_width, 
              [
                [
                    object_class, # 物体类型 为数字
                    object_xmin,  # box位置
                    object_ymin, 
                    object_xmax, 
                    object_ymax
                ],...
              ],...
            ],...
        ]
        """
        self.data_set = []
        self.object_names_to_index = get_object_to_index(dict_class_index_file)
        self.xml_dir = xml_dir
        self.image_dir = image_dir
        self.transform_size = transform_size
        self.grid_num = grid_num
        # 读取所有的xml文件
        list_file_name = os.listdir(self.xml_dir)
        # 分析内容
        for file_name in list_file_name:
            xml_info = self.xml_parse(os.path.join(xml_dir, file_name))
            self.data_set.append(xml_info)

    def __len__(self):
        """获取数据集数量"""
        return len(self.data_set)

    def __getitem__(self, item):
        """
           TODO:
           1. 按顺序依次取出第item个训练数据img及其对应的样本标签label
           2. 图像数据要进行预处理，并最终转换为(c, h, w)的维度，同时转换为torch.tensor
           3. 样本标签要按需要转换为指定格式的torch.tensor
           :return
           img:
           labels:tensor:[[(x,y,w,h,c)*2,class*20],]
           x:y为所在cell的偏移量
        """
        label = self.data_set[item]

        # 打开图片
        image = cv2.imread(os.path.join(self.image_dir, label[0]))
        image = cv2.resize(image, (self.transform_size, self.transform_size))
        transform = transforms.Compose([transforms.ToTensor(), ])
        img = transform(image)

        labels = torch.tensor(label[3])  # [[class,x1,y1,x2,y2],...]
        # label需要转换为 7*7*30的格式
        labels = self._transform_label(labels)  #
        return img, labels

    def _transform_label(self, labels):
        """

        :param labels: [[class,x1,y1,x2,y2],...]
        :return: 7*7*[类型，bias_x，bias_y,w,h]
        """
        result = torch.zeros((self.grid_num, self.grid_num, 30))
        cw = 1 / self.grid_num  #
        wh = labels[:, 3:] - labels[:, 1:3]  # 宽高
        xy = (labels[:, 3:] + labels[:, 1:3]) / 2  # 中心点位置

        for i in range(xy.size()[0]):
            ij = (xy[i] / cw).floor() - 1  # ij:xy 第几个cell
            ij[ij < 0] = 0  # 超过了则设置为0
            result[int(ij[0]), int(ij[1]), 4] = 1  # c设置为1
            result[int(ij[0]), int(ij[1]), 9] = 1
            result[int(ij[0]), int(ij[1]), int(labels[i][0]) + 9] = 1  # class_index 从第一位开始

            # 按百分比保存
            _xy = xy[i]  #
            _wh = wh[i]  #

            result[int(ij[0]), int(ij[1]), 2:4] = _wh
            result[int(ij[0]), int(ij[1]), :2] = _xy
            result[int(ij[0]), int(ij[1]), 7:9] = _wh
            result[int(ij[0]), int(ij[1]), 5:7] = _xy

        return result

    def xml_parse(self, file_path):
        """

        :param file_path:
        :return:
        """
        dom = xml.dom.minidom.parse(file_path)
        # 得到文档元素对象
        root = dom.documentElement
        # 文件名
        filename = root.getElementsByTagName('filename')[0].childNodes[0].nodeValue
        # 目标照片大小
        file_height = int(root.getElementsByTagName('height')[0].childNodes[0].nodeValue)
        file_width = int(root.getElementsByTagName('width')[0].childNodes[0].nodeValue)
        # 文件内物体的类型长宽高中心位置
        objects = []
        #
        object_tags = root.getElementsByTagName('object')
        for tag in object_tags:
            object_class = self.object_names_to_index[tag.getElementsByTagName('name')[0].childNodes[0].nodeValue]
            if object_class is None:
                raise
            # 左上角为起点
            object_xmin = int(tag.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)
            object_ymin = int(tag.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)
            object_xmax = int(tag.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
            object_ymax = int(tag.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)

            # 按百分比保存
            object_xmin = object_xmin / file_width
            object_ymin = object_ymin / file_height
            object_xmax = object_xmax / file_width
            object_ymax = object_ymax / file_height

            objects.append([object_class, object_xmin, object_ymin, object_xmax, object_ymax])

        return [filename, file_width, file_height, objects]

    def paint_xml_img(self, image_name):
        for d in self.data_set:
            if d[0] == image_name:
                image = cv2.imread(os.path.join(self.image_dir, image_name))
                for p in d[3]:
                    left_up = (int(p[1] * d[1]), int(p[2] * d[2]))
                    right_bottom = (int(p[3] * d[1]), int(p[4] * d[2]))
                    cv2.rectangle(image, left_up, right_bottom, (255, 0, 0), 2)
                    cv2.putText(image, "%s" % (self.get_name_by_index(p[0])), left_up, cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (255, 255, 255), 1, 8)
            cv2.imshow(image_name, image)
            cv2.waitKey(0)

            # ii = np.transpose(data[0], (1, 2, 0))
            # plt.imshow(ii)
            # plt.show()
            return

    def get_name_by_index(self, index):
        d = self.object_names_to_index
        for i in d:
            if d[i] == index:
                return i
        return None

def print_index_of_class(xml_dir):
    """
    获取当前所有的xml的类型，以及对应的数字类型
    :param xml_dir:
    :return:

    """
    d = {}

    list_file_name = os.listdir(xml_dir)
    for file_name in list_file_name:
        file_path = os.path.join(xml_dir, file_name)
        dom = xml.dom.minidom.parse(file_path)
        # 得到文档元素对象
        root = dom.documentElement
        object_tags = root.getElementsByTagName('object')
        for tag in object_tags:
            object_name = tag.getElementsByTagName('name')[0].childNodes[0].nodeValue
            if object_name in d:
                continue
            else:
                d[object_name] = (len(d) + 1)
    print(d)


def get_object_to_index(file_path) -> dict:
    """
    获取文件下物体类型以及对应的数字
    :param file_path:
    :return:{name:index}
    """
    result = {}
    with open(file_path) as r:
        lines = r.readlines()
        for line in lines:
            line = line.strip()  #
            i = line.split(":")
            result[i[0]] = int(i[1])
    return result


def get_name_by_index(file_path, index) -> dict:
    """
    :param file_path:
    :param index:
    :return: index对应的名称
    """
    d = get_object_to_index(file_path)
    for i in d:
        if d[i] == index:
            return i
    return None


def loop_helper(loop_param_list, start_fun=None, excute_fun=None, end_fun=None):
    """
    loop_param:
    [[1,{}],[3,{}]....]
    """
    if start_fun is not None:
        start_fun()

    for i in range(len(loop_param_list)):
        loop_param = loop_param_list[i]

        for k in range(int(loop_param[0])):
            if excute_fun is not None:
                excute_fun(*[i, k, loop_param[1]])

    if end_fun is not None:
        excute_fun()


if __name__ == '__main__':
    # 测试
    root_path = PROJECT_ROOT_PATH
    xml_dir = os.path.join(root_path, r'data\train\VOC2007\Annotations')
    image_dir = os.path.join(root_path, r'data\train\VOC2007\JPEGImages')
    classname_to_index_file_path = os.path.join(root_path, r'data\voc2007-class-to-index.txt')
    minist_dir = os.path.join(root_path, r'data')


    # 1：根据index获取其对应名称
    # get_name_by_index(classname_to_index_file_path, 1)

    # 2:获取物体类型以及对应的数字种类
    # r = get_object_to_index(classname_to_index_file_path)
    # print(r)

    # 3:读取xml文档的信息
    # a = YoloDataset(xml_dir, image_dir, classname_to_index_file_path, 448)
    # a.paint_xml_img("000005.jpg")

    # 4:MINIST数据集
    # a = MnistDataset()

    # 5:
    # def start(*k):
    #     print(k)
    #
    # def excute(*k):
    #     print(k)
    #
    # def end(*k):
    #     print(k)
    #
    # loop_param_list = [[1, {"1": 1}], [2, {"3": 1}]]
    # loop_helper(loop_param_list, start_fun=start, excute_fun=excute, end_fun=end)
    pass
