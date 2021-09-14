import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class YoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def compute_iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

    # 可以参考下面的代码 实现计算iou的tensor代码
    def _calculate_iou(self, bbox1, bbox2):
        """
        :param bbox1: [x,y,w,h]
        :param bbox2: [x,y,w,h]
        :return:
        """
        # x1,y1.x2,y2
        x1, y1, x2, y2 = bbox1[0] - bbox1[2] / 2, bbox1[1] - bbox1[3] / 2, bbox1[0] + bbox1[2] / 2, bbox1[1] + bbox1[
            3] / 2
        x3, y3, x4, y4 = bbox2[0] - bbox2[2] / 2, bbox2[1] - bbox2[3] / 2, bbox2[0] + bbox2[2] / 2, bbox2[1] + bbox2[
            3] / 2

        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)

        inter_section = max(0, min(x2, x4) - max(x1, x3)) * max(0, min(y2, y4) - max(y1, y3))
        return inter_section / (area1 + area2 - inter_section)

    def forward(self, pred_tensor, target_tensor):
        # input:[batch,7,7,30]
        """
            preds : (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c]
            target : (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c]
         关于最后的shape30 其真正的含义为,2*(B*5+C)
         B:(px,py,pw,ph,置信度) ,
         C:预测概率,在此训练中数量为20个
         px:特指为在cell中,以cell左下角为起点的x偏移量
         py:特指为在cell中,以cell左下角为起点的y偏移量

        """
        # batchsize
        N = pred_tensor.size()[0]
        # coo 表示存在物体
        coo_mask = target_tensor[:, :, :, 4] > 0  # (batchsize,S,S) true or false
        # noo 不存在物体
        noo_mask = target_tensor[:, :, :, 4] == 0

        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)  # (batchsize,S,S,30)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)

        # 获取所有存在对象的预测cell
        coo_pred = pred_tensor[coo_mask].view(-1, 30)
        box_pred = coo_pred[:, :10].contiguous().view(-1, 5)
        class_pred = coo_pred[:, 10:]

        # 获取真实存在对象的cell信息
        coo_target = target_tensor[coo_mask].view(-1, 30)
        box_target = coo_target[:, :10].contiguous().view(-1, 5)
        class_target = coo_target[:, 10:]

        # 不存在对象的预测cell
        noo_pred = pred_tensor[noo_mask].view(-1, 30)
        noo_target = target_tensor[noo_mask].view(-1, 30)
        #
        noo_pred_mask = torch.zeros(noo_pred.size(), dtype=torch.bool)
        # noo_pred_mask.zero_()
        noo_pred_mask[:, 4] = 1
        noo_pred_mask[:, 9] = 1
        noo_pred_mask[noo_pred_mask == 1] = True
        noo_pred_mask[noo_pred_mask == 0] = False
        noo_pred_mask = noo_pred_mask.type(torch.bool)
        # 获取所有不含目标的cell的两个box的置信度
        noo_pred_c = noo_pred[noo_pred_mask]
        noo_target_c = noo_target[noo_pred_mask]
        # 不含目标的置信度损失
        nooobj_loss = F.mse_loss(noo_pred_c, noo_target_c, size_average=False)

        # compute contain obj loss
        coo_response_mask = torch.zeros(box_target.size())
        coo_not_response_mask = torch.zeros(box_target.size())
        box_target_iou = torch.zeros(box_target.size())
        # choose the best iou box
        for i in range(0, box_target.size()[0], 2):
            box1 = box_pred[i:i + 2]  # 包含目标的两个box
            box1_xyxy = Variable(torch.FloatTensor(box1.size()))
            # x,y
            box1_xyxy[:, :2] = box1[:, :2] - box1[:, 2:4] / 2  # x1,y1
            box1_xyxy[:, 2:4] = box1[:, :2] + box1[:, 2:4] / 2  # x2,y2
            box2 = box_target[i].view(-1, 5)  # box_target 是两个一摸一样的box
            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:, :2] = box2[:, :2] - box2[:, 2:4] / 2
            box2_xyxy[:, 2:4] = box2[:, :2] + box2[:, 2:4] / 2
            iou = self.compute_iou(box1_xyxy[:, :4], box2_xyxy[:, :4])  # [2,1]
            # 找最大的IOU 0 或者 1
            max_iou, max_index = iou.max(0)
            # max_index = max_index.data.cuda()
            coo_response_mask[i + max_index] = 1
            coo_not_response_mask[i + 1 - max_index] = 1

            #####
            # we want the confidence score to equal the
            # intersection over union (IOU) between the predicted box
            # and the ground truth
            #####
            box_target_iou[i + max_index, 4] = max_iou
            # 这里训练使用1，效果也差不多
            # box_target_iou[i + max_index, 4] = 1
        box_target_iou = Variable(box_target_iou)
        # 置信度的损失函数
        # 1.response loss
        coo_response_mask = coo_response_mask.type(torch.bool)
        box_pred_response = box_pred[coo_response_mask].view(-1, 5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1, 5)
        box_target_response = box_target[coo_response_mask].view(-1, 5)
        contain_loss = F.mse_loss(box_pred_response[:, 4], box_target_response_iou[:, 4], size_average=False)
        loc_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], size_average=False) + F.mse_loss(
            torch.sqrt(box_pred_response[:, 2:4]), torch.sqrt(box_target_response[:, 2:4]), size_average=False)
        # 2.not response loss
        coo_not_response_mask = coo_not_response_mask.type(torch.bool)
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1, 5)
        box_target_not_response = box_target[coo_not_response_mask].view(-1, 5)
        box_target_not_response[:, 4] = 0

        # I believe this bug is simply a typo
        not_contain_loss = F.mse_loss(box_pred_not_response[:, 4], box_target_not_response[:, 4], size_average=False)

        # 3.class loss
        class_loss = F.mse_loss(class_pred, class_target, size_average=False)

        return (
                           self.l_coord * loc_loss + 2 * contain_loss + not_contain_loss + self.l_noobj * nooobj_loss + class_loss) / N

    def compute_iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou
