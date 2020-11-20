import torch.nn as nn
from network.resnet import resnet18
from network.rpn import RPN
from network.neck import AdjustLayer


class buildlModel(nn.Module):
    def __init__(self):
        super(buildlModel, self).__init__()
        self.backbone=resnet18(pretrained=True)
        self.neck=AdjustLayer()
        self.rpn=RPN()

    def forward(self,data):
        template = data['template']
        search = data['search']

        tempFeatList=self.backbone(template)
        searFeatList = self.backbone(search)
        tempFeatList=self.neck(tempFeatList)
        searFeatList=self.neck(searFeatList)
        cls,loc=self.rpn(tempFeatList,searFeatList)

        return cls,loc