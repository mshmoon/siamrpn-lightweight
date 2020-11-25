import torch.nn as nn
import torch.nn.functional as F

class Fpn(nn.Module):
    def __init__(self):
        super(Fpn, self).__init__()
        self.upLayer1=nn.Sequential(
            nn.Conv2d(512,256,1,1,0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.upLayer2 = nn.Sequential(
            nn.Conv2d(256, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.upLayer3 = nn.Sequential(
            nn.Conv2d(128, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.center_size=7

    def forward(self,x):

        x0 = self.upLayer1(x[0])
        x1 = self.upLayer2(x[1])
        x2 = self.upLayer3(x[2])

        x1=F.upsample_bilinear(x0,scale_factor=2)+x1
        x2=F.upsample_bilinear(x1,scale_factor=2)+x2
        if x2.size(3) < 20:
            l = (x2.size(3) - self.center_size) // 2
            r = l + self.center_size
            x2 = x2[:, :, l:r, l:r]
        else:
            x2=F.upsample_bilinear(x1,size=(31,31))
        x=x2
        return x


class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=7):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            )
        self.center_size = center_size
        self.fpn=Fpn()

    def forward(self, x):
        x=self.fpn(x)
        x = self.downsample(x)
        return x