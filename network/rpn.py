import torch.nn as nn

class RPN(nn.Module):
    def __init__(self,inChanels,outChanels,layer):
        super(RPN,self).__init__()

        self.inChanels = inChanels
        self.outChanels = outChanels
        self.layer = layer

        self.rpnHead=nn.Sequential(
            nn.Conv2d(self.inChanels,self.outChanels,1,1,0),
            nn.BatchNorm2d(self.outChanels),
            nn.ReLU(inplace=True)
        )

        assert layer>1,"layer number must grater to one."

        self.convBody=nn.ModuleList([nn.Conv2d(self.outChanels,self.outChanels) for i in range(self.layer-1)])
        self.bnBody = nn.ModuleList([nn.batchNorm2d(self.outChanels) for i in range(self.layer - 1)])
        self.relu=nn.ReLU(inplace=True)

    def forward(self,x,y):
        x=self.rpnHead(x)
        for convBn in zip(self.convBody,self.bnBody):
            x=convBn[0](x)
            x=convBn[1](x)
            x=self.relu(x)
        return x
