import math
import numpy as np
import torch
import torch.nn as nn


def calculate_iou(_box_a, _box_b):
    # -----------------------------------------------------------#
    #   计算真实框的左上角和右下角
    # -----------------------------------------------------------#
    b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
    b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
    # -----------------------------------------------------------#
    #   计算先验框获得的预测框的左上角和右下角
    # -----------------------------------------------------------#
    b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
    b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2

    # -----------------------------------------------------------#
    #   将真实框和预测框都转化成左上角右下角的形式
    # -----------------------------------------------------------#
    box_a = torch.zeros_like(_box_a)
    box_b = torch.zeros_like(_box_b)
    box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
    box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2

    # -----------------------------------------------------------#
    #   A为真实框的数量，B为先验框的数量
    # -----------------------------------------------------------#
    A = box_a.size(0)
    B = box_b.size(0)

    # -----------------------------------------------------------#
    #   计算交的面积
    # -----------------------------------------------------------#
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    inter = inter[:, :, 0] * inter[:, :, 1]
    # -----------------------------------------------------------#
    #   计算预测框和真实框各自的面积
    # -----------------------------------------------------------#
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    # -----------------------------------------------------------#
    #   求IOU
    # -----------------------------------------------------------#
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def clip_by_tensor(t, t_min, t_max):
    t = t.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def MSELoss(pred, target):
    return torch.pow(pred - target, 2)


def BCELoss(pred, target):
    epsilon = 1e-7
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_class, input_shape, device, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]):
        super().__init__()
        self.anchors = anchors
        self.num_class = num_class
        self.bbox_attrs = 5 + num_class
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.ignore_threshold = 0.5
        self.device = device

    def forward(self, l, input, targets=None):
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)

        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w

        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        prediction = input.view(bs, len(self.anchors_mask[l]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4,
                                                                                                    2).contiguous()
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])

        w = prediction[..., 2]
        h = prediction[..., 3]

        conf = torch.sigmoid(prediction[..., 4])

        pred_cls = torch.sigmoid(prediction[..., 5:])

        y_true, noobj_mask, box_loss_scale = self.get_target(l, targets, scaled_anchors, in_h, in_w)

        noobj_mask = self.get_ignore(l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask)

        y_true.to(self.device)
        noobj_mask.to(self.device)
        box_loss_scale.to(self.device)

        box_loss_scale = 2 - box_loss_scale

        # -----------------------------------------------------------#
        #   计算中心偏移情况的loss，使用BCELoss效果好一些
        # -----------------------------------------------------------#
        loss_x = torch.sum(BCELoss(x, y_true[..., 0]) * box_loss_scale * y_true[..., 4])
        loss_y = torch.sum(BCELoss(y, y_true[..., 1]) * box_loss_scale * y_true[..., 4])
        # -----------------------------------------------------------#
        #   计算宽高调整值的loss
        # -----------------------------------------------------------#
        loss_w = torch.sum(MSELoss(w, y_true[..., 2]) * 0.5 * box_loss_scale * y_true[..., 4])
        loss_h = torch.sum(MSELoss(h, y_true[..., 3]) * 0.5 * box_loss_scale * y_true[..., 4])
        # -----------------------------------------------------------#
        #   计算置信度的loss
        # -----------------------------------------------------------#
        loss_conf = torch.sum(BCELoss(conf, y_true[..., 4]) * y_true[..., 4]) + \
                    torch.sum(BCELoss(conf, y_true[..., 4]) * noobj_mask)

        loss_cls = torch.sum(BCELoss(pred_cls[y_true[..., 4] == 1], y_true[..., 5:][y_true[..., 4] == 1]))

        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        num_pos = torch.sum(y_true[..., 4])
        num_pos = torch.max(num_pos, torch.ones_like(num_pos))
        return loss, num_pos

    def get_target(self, l, targets, anchors, in_h, in_w):
        # -----------------------------------------------------#
        #   计算一共有多少张图片
        # -----------------------------------------------------#
        bs = len(targets)
        # -----------------------------------------------------#
        #   用于选取哪些先验框不包含物体
        # -----------------------------------------------------#
        noobj_mask = torch.ones(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)
        # -----------------------------------------------------#
        #   让网络更加去关注小目标
        # -----------------------------------------------------#
        box_loss_scale = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)
        # -----------------------------------------------------#
        #   batch_size, 3, 13, 13, 5 + num_classes
        # -----------------------------------------------------#
        y_true = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, self.bbox_attrs, requires_grad=False)
        for b in range(bs):
            if len(targets[b]['boxes']) == 0:
                continue
            batch_target = torch.zeros_like(targets[b]['boxes'])
            # -------------------------------------------------------#
            #   计算出正样本在特征层上的中心点
            # -------------------------------------------------------#
            batch_target[:, [0, 2]] = targets[b]['boxes'][:, [0, 2]] * in_w
            batch_target[:, [1, 3]] = targets[b]['boxes'][:, [1, 3]] * in_h
            # batch_target[:, 4] = targets[b]['labels']
            batch_target = batch_target.cpu()

            # -------------------------------------------------------#
            #   将真实框转换一个形式
            #   num_true_box, 4
            # -------------------------------------------------------#
            gt_box = torch.FloatTensor(torch.cat((torch.zeros((batch_target.size(0), 2)), batch_target[:, 2:4]), 1))
            # -------------------------------------------------------#
            #   将先验框转换一个形式
            #   9, 4
            # -------------------------------------------------------#
            anchor_shapes = torch.FloatTensor(
                torch.cat((torch.zeros((len(anchors), 2)), torch.FloatTensor(anchors)), 1))
            # -------------------------------------------------------#
            #   计算交并比
            #   self.calculate_iou(gt_box, anchor_shapes) = [num_true_box, 9]每一个真实框和9个先验框的重合情况
            #   best_ns:
            #   [每个真实框最大的重合度max_iou, 每一个真实框最重合的先验框的序号]
            # -------------------------------------------------------#
            best_ns = torch.argmax(calculate_iou(gt_box, anchor_shapes), dim=-1)

            for t, best_n in enumerate(best_ns):
                if best_n not in self.anchors_mask[l]:
                    continue
                # ----------------------------------------#
                #   判断这个先验框是当前特征点的哪一个先验框
                # ----------------------------------------#
                k = self.anchors_mask[l].index(best_n)
                # ----------------------------------------#
                #   获得真实框属于哪个网格点
                # ----------------------------------------#
                i = torch.floor(batch_target[t, 0]).long()
                j = torch.floor(batch_target[t, 1]).long()
                # ----------------------------------------#
                #   取出真实框的种类
                # ----------------------------------------#
                c = targets[b]['labels'][t].long()
                # ----------------------------------------#
                #   noobj_mask代表无目标的特征点
                # ----------------------------------------#
                noobj_mask[b, k, j, i] = 0
                # ----------------------------------------#
                #   tx、ty代表中心调整参数的真实值
                # ----------------------------------------#
                y_true[b, k, j, i, 0] = batch_target[t, 0] - i.float()
                y_true[b, k, j, i, 1] = batch_target[t, 1] - j.float()
                y_true[b, k, j, i, 2] = math.log(batch_target[t, 2] / anchors[best_n][0])
                y_true[b, k, j, i, 3] = math.log(batch_target[t, 3] / anchors[best_n][1])
                y_true[b, k, j, i, 4] = 1
                y_true[b, k, j, i, c + 5] = 1
                # ----------------------------------------#
                #   用于获得xywh的比例
                #   大目标loss权重小，小目标loss权重大
                # ----------------------------------------#
                box_loss_scale[b, k, j, i] = batch_target[t, 2] * batch_target[t, 3] / in_w / in_h
        return y_true, noobj_mask, box_loss_scale

    def get_ignore(self, l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask):
        # -----------------------------------------------------#
        #   计算一共有多少张图片
        # -----------------------------------------------------#
        bs = len(targets)

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        # -----------------------------------------------------#
        #   生成网格，先验框中心，网格左上角
        # -----------------------------------------------------#
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(y.shape).type(FloatTensor)

        # 生成先验框的宽高
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        anchor_w = FloatTensor(scaled_anchors_l).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors_l).index_select(1, LongTensor([1]))

        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        # -------------------------------------------------------#
        #   计算调整后的先验框中心与宽高
        # -------------------------------------------------------#
        pred_boxes_x = torch.unsqueeze(x.data + grid_x, -1)
        pred_boxes_y = torch.unsqueeze(y.data + grid_y, -1)
        pred_boxes_w = torch.unsqueeze(torch.exp(w.data) * anchor_w, -1)
        pred_boxes_h = torch.unsqueeze(torch.exp(h.data) * anchor_h, -1)
        pred_boxes = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim=-1)

        for b in range(bs):
            # -------------------------------------------------------#
            #   将预测结果转换一个形式
            #   pred_boxes_for_ignore      num_anchors, 4
            # -------------------------------------------------------#
            pred_boxes_for_ignore = pred_boxes[b].view(-1, 4)
            # -------------------------------------------------------#
            #   计算真实框，并把真实框转换成相对于特征层的大小
            #   gt_box      num_true_box, 4
            # -------------------------------------------------------#
            if len(targets[b]['boxes']) > 0:
                batch_target = torch.zeros_like(targets[b]['boxes'])
                # -------------------------------------------------------#
                #   计算出正样本在特征层上的中心点
                # -------------------------------------------------------#
                batch_target[:, [0, 2]] = targets[b]['boxes'][:, [0, 2]] * in_w
                batch_target[:, [1, 3]] = targets[b]['boxes'][:, [1, 3]] * in_h
                batch_target = batch_target[:, :4]
                # -------------------------------------------------------#
                #   计算交并比
                #   anch_ious       num_true_box, num_anchors
                # -------------------------------------------------------#
                anch_ious = calculate_iou(batch_target, pred_boxes_for_ignore)
                # -------------------------------------------------------#
                #   每个先验框对应真实框的最大重合度
                #   anch_ious_max   num_anchors
                # -------------------------------------------------------#
                anch_ious_max, _ = torch.max(anch_ious, dim=0)
                anch_ious_max = anch_ious_max.view(pred_boxes[b].size()[:3])
                noobj_mask[b][anch_ious_max > self.ignore_threshold] = 0
        return noobj_mask
