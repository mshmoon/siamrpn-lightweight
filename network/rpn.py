import torch.nn as nn
import torch.nn.functional as F

class Template(nn.Module):
    def __init__(self,inChanels,outChanels,layer):
        super(Template,self).__init__()

        self.inChanels = inChanels
        self.outChanels = outChanels
        self.layer = layer

        self.rpnHead=nn.Sequential(
            nn.Conv2d(self.inChanels,self.outChanels,1,1,0),
            nn.BatchNorm2d(self.outChanels),
            nn.ReLU(inplace=True)
        )

        assert layer>=1,"layer number must grater to one."

        self.convBody=nn.ModuleList([nn.Conv2d(self.inChanels,self.inChanels,3,1,1) for i in range(self.layer-1)])
        self.convBody.append(nn.Conv2d(self.inChanels,self.outChanels,3,1,0))

        self.bnBody = nn.ModuleList([nn.batchNorm2d(self.inChanels) for i in range(self.layer - 1)])
        self.bnBody.append(nn.BatchNorm2d(outChanels))

        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):

        for convBn in zip(self.convBody,self.bnBody):
            x=convBn[0](x)
            x=convBn[1](x)
            x=self.relu(x)
        return x

class Corr(nn.Module):
    def __init__(self):
        super(Corr, self).__init__()

    def forward(self,x,kernel):

        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch * channel, x.size(2), x.size(3))
        kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch * channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out

class RPN(nn.Module):
    def __init__(self,in_channels=256,out_channels=256,layer=1):
        super(RPN, self).__init__()
        self.loc = Template(in_channels,out_channels,layer)
        self.cls = Template(in_channels,out_channels,layer)
        self.loc_tail = nn.Conv2d(256,20,1,1,0)
        self.cls_tail = nn.Conv2d(256, 10, 1, 1, 0)
        self.corr=Corr()

    def forward(self, x, y):
        x_loc = self.loc(x)
        y_loc = self.loc(y)

        x_cls = self.cls(x)
        y_cls = self.cls(y)
        loc = self.loc_tail(self.corr(y_loc,x_loc))
        cls = self.cls_tail(self.corr(y_cls, x_cls))

        return cls,loc