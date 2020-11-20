import torch
import torch.nn as nn
import torch.functional as F

class LogSoftMax(nn.Module):
    def __init__(self):
        super(LogSoftMax, self).__init__()

    def forward(self,cls):
        b,a2,h,w=cls.size()
        cls=cls.view(b,2,a2//2,h,w)
        cls=cls.permute(0,2,3,4,1).contiguous()
        cls=F.log_softMax(cls,dim=4)
        return cls

class SelectCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SelectCrossEntropyLoss, self).__init__()

    def get_cls_loss(self,pred, label, select):
        if len(select.size()) == 0 or \
                select.size() == torch.Size([0]):
            return 0
        pred = torch.index_select(pred, 0, select)
        label = torch.index_select(label, 0, select)
        return F.nll_loss(pred, label)

    def select_cross_entropy_loss(self,pred, label):
        pred = pred.view(-1, 2)
        label = label.view(-1)
        pos = label.data.eq(1).nonzero().squeeze()
        neg = label.data.eq(0).nonzero().squeeze()
        loss_pos = self.get_cls_loss(pred, label, pos)
        loss_neg = self.get_cls_loss(pred, label, neg)
        return loss_pos * 0.5 + loss_neg * 0.5

    def forward(self,pred, label):
        loss=self.select_cross_entropy_loss(pred, label)

        return loss

class WeightL1Loss(nn.Module):
    def __init__(self):
        super(WeightL1Loss, self).__init__()

    def forward(self,pred_loc, label_loc, loss_weight):
        b, _, sh, sw = pred_loc.size()
        pred_loc = pred_loc.view(b, 4, -1, sh, sw)
        diff = (pred_loc - label_loc).abs()
        diff = diff.sum(dim=1).view(b, -1, sh, sw)
        loss = diff * loss_weight
        return loss.sum().div(b)