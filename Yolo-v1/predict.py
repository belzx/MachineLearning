import os

import cv2
import torch
import torchvision.transforms as transforms

import model
import utils


def predict():
    if predict_type == 0:
        image_predice()
    else:
        # TODO
        raise


def image_predice():
    net = model.YoloV1Net(load_model_path=load_model_path)
    # 加载图片
    image = cv2.imread(source_path)
    # 获取宽高
    w = image.shape[1]
    h = image.shape[0]
    # 调整为(448*448*3)
    img = cv2.resize(image, (448, 448))
    transform = transforms.Compose([transforms.ToTensor(), ])
    # 转tensor调整为(3*448 * 448)
    img = transform(img)
    img = img.unsqueeze(0)
    # 预测
    pred = net(img)
    # 处理分析结果
    boxes, cls_indexs, probs = decoder(pred, w, h)
    result = []
    for i, box in enumerate(boxes):
        x1 = int(box[0].item())
        y1 = int(box[1].item())

        x2 = int(box[2].item())
        y2 = int(box[3].item())

        cls_index = cls_indexs[i]
        cls_index = int(cls_index)  # convert LongTensor to int
        prob = probs[i]
        prob = float(prob)

        result.append([(x1, y1), (x2, y2), cls_index, prob])
    #
    for left_up, right_bottom, class_index, prob in result:
        cv2.rectangle(image, left_up, right_bottom, (255, 0, 0), 2)
        cv2.putText(image,
                    "%s %s" % (utils.get_name_by_index(classname_to_index_file_path, class_index + 1), str(prob)[:4]),
                    left_up, cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, 8)

    cv2.imshow("result", image)
    cv2.waitKey(0)
    if out_path is not None:
        # 结果输出
        cv2.imwrite(out_path, image)


def decoder(pred, w, h):
    '''
    pred (tensor) 1x7x7x30
    return (tensor) box[[x1,y1,x2,y2]] label[...]
    '''
    boxes = []
    cls_indexs = []
    probs = []
    pred = pred.squeeze(0)  # 7x7x30

    # 两个预测框的置信度
    contain1 = pred[:, :, 4].unsqueeze(2)
    contain2 = pred[:, :, 9].unsqueeze(2)
    #
    contain = torch.cat((contain1, contain2), 2)
    # 置信度至少大于0.5
    # mask1 = contain > 0.5  # 大于阈值
    _, mask = contain.max(2)  # we always select the best contain_prob what ever it>0.9
    # 大于阈值 且为最大
    # gt 大于0
    # mask = (mask1 + mask2).gt(0)
    for i in range(7):
        for j in range(7):
            # 1:保留最大置信度的框
            b = mask[i, j].item()
            if _[i, j].item() > c_threshold:  # 置信度
                box = pred[i, j, b * 5:b * 5 + 4]
                contain_prob = torch.FloatTensor([pred[i, j, b * 5 + 4]])

                # 由百分比还原真正的长宽
                xy = box[0:2] * torch.tensor([w, h])
                wh = box[2:4] * torch.tensor([w, h])
                box_xy = torch.FloatTensor(box.size())
                box_xy[:2] = xy - wh / 2
                box_xy[2:4] = xy + wh / 2
                max_prob, cls_index = torch.max(pred[i, j, 10:], 0)

                if float((contain_prob * max_prob)[0]) > c_class_threshold:
                    cls_indexs.append(torch.tensor([1.0, ]) * cls_index)
                    probs.append(contain_prob * max_prob)
                    boxes.append(box_xy)

    if len(boxes) == 0:
        boxes = torch.zeros((1, 4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
    else:
        boxes = torch.cat(boxes, 0).view(-1, 4)  # (n,4)
        probs = torch.cat(probs, 0).view(-1, 1)  # (n,)
        cls_indexs = torch.cat(cls_indexs, 0).view(-1, 1)  # (n,)
    # 概率大于0.5
    keep = nms(boxes, probs)
    return boxes[keep], cls_indexs[keep], probs[keep]


def nms(bboxes, scores, threshold=0.5):
    '''
    TODO
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    '''
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    # _:数据 order:下标列表
    _, order = scores.sort(0, descending=True)
    keep = []

    for k in range(order.numel()):
        i = order[k]
        keep.append(i)

    # order.numel()的长度
    # while order.numel() > 0:
    #     i = order[0]
    #     keep.append(i)
    #     break
    #     if order.numel() == 1:
    #         break

    # xx1 = x1[order[1:]].clamp(min=x1[i])
    # yy1 = y1[order[1:]].clamp(min=y1[i])
    # xx2 = x2[order[1:]].clamp(max=x2[i])
    # yy2 = y2[order[1:]].clamp(max=y2[i])
    #
    # w = (xx2 - xx1).clamp(min=0)
    # h = (yy2 - yy1).clamp(min=0)
    # inter = w * h
    #
    # ovr = inter / (areas[i] + areas[order[1:]] - inter)
    # ids = (ovr <= threshold).nonzero().squeeze()
    # if ids.numel() == 0:
    #     break
    # order = order[ids + 1]
    return torch.LongTensor(keep)


if __name__ == '__main__':
    root_path = utils.PROJECT_ROOT_PATH
    # 模型路径
    load_model_path = os.path.join(root_path, r'Yolo-v1\output\models\best.pth')
    # 资源路径
    source_path = os.path.join(root_path, r'data\train\VOC2007\JPEGImages\000005.jpg')
    # 类别对应下标
    classname_to_index_file_path = os.path.join(root_path, r'data\voc2007-class-to-index.txt')
    # 识别模式 0:照片，1：视频
    predict_type = 0
    # 置信度閾值
    c_threshold = 0.2
    # 置信度*class_prod閾值
    c_class_threshold = 0.1
    # 文件输出位置,为None则不保存,只展示
    out_path = os.path.join(root_path, r'Yolo-v1\output\image\result.png')
    print("START!")
    predict()
    print("END")
