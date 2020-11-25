import torch.nn as nn
from network.resnet import resnet18
from network.rpn import RPN
from network.neck import AdjustLayer


class buildlModel(nn.Module):
    """
    function:共下采样3次
    """
    def __init__(self,mode):
        super(buildlModel, self).__init__()
        self.backbone=resnet18(pretrained=True)
        self.neck=AdjustLayer(256,256)
        self.rpn=RPN(256,256,1)
        self.mode=mode

    def template(self, z):
        zf = self.backbone(z)
        self.zf = self.neck(zf)

    def track(self, x):
        xf = self.backbone(x)
        xf = self.neck(xf)
        cls, loc = self.rpn(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc,
                'mask': None
               }

    def forward(self,data):
        if self.mode=="train":
            template = data['template']
            search = data['search']

            tempFeatList=self.backbone(template)
            searFeatList = self.backbone(search)
            tempFeat=self.neck(tempFeatList)
            searFeat=self.neck(searFeatList)
            cls,loc=self.rpn(tempFeat,searFeat)

            return cls,loc