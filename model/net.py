import math
import torch
from torch import nn
import torch.nn.functional as F
from model.network import BSConvU, CCALayer, ESA, Baseblock_
# 315 32.3230      38.52



    
class Block(nn.Module):
    def __init__(self, nc,conv=BSConvU, ratio=0.25):
        super(Block, self).__init__()
        self.ratio = ratio
        f = int(nc * ratio)
        self.dc = self.distilled_channels = nc // 2
        self.rc = self.remaining_channels = nc
        self.c1_d = nn.Conv2d(nc, self.dc, 1)
        self.c1_r = conv(nc, f, kernel_size=3, )
        self.c2_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c2_r = conv(self.remaining_channels, f )
        self.c3_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c3_r = conv(self.remaining_channels,f  )

        self.c4 = conv(self.remaining_channels,f)
        self.act = nn.GELU()
        self.c5 = nn.Conv2d(self.dc * 3 + self.rc, nc, 1)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 +input )

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2 +r_c1 )

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3 +r_c2)

        r_c4 = self.act(self.c4(r_c3))
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.c5(out)
        return out

class Susi0(nn.Module):
    def __init__(self, nc,conv=Block, ratio=0.25):
        super(Susi0, self).__init__()

        self.ratio = ratio
        self.dc = self.distilled_channels = nc // 2
        self.rc = self.remaining_channels = nc

        self.c1_d = nn.Conv2d(nc, self.dc, 1)
        self.c1_r = conv(nc, BSConvU,  ratio=self.ratio)

        self.c2_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c2_r = conv(nc, BSConvU,  ratio=self.ratio)
        #
        self.c3_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c3_r = conv(nc, BSConvU,  ratio=self.ratio)

        self.c4_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        # self.c4_r = conv(nc, BSConvU, ratio=self.ratio)
        # #
        self.c5_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        # self.c5_r = conv(nc, BSConvU, ratio=self.ratio)
        #
        self.c6_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        # self.c6_r = conv(nc, BSConvU, ratio=self.ratio)

        self.c7_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        # self.c4_r = conv(nc, BSConvU, ratio=self.ratio)
        # #
        self.c8_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        # self.c5_r = conv(nc, BSConvU, ratio=self.ratio)
        #
        self.c9_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        # self.c6_r = conv(nc, BSConvU, ratio=self.ratio)

        self.c4 = conv(nc, BSConvU, ratio=self.ratio)
        self.act = nn.GELU()
        self.c5 = nn.Conv2d(self.dc * 6 +self.rc, nc, 1)
        self.esa = ESA(nc, Baseblock_,ratio)
        self.cca = CCALayer(nc)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1+input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2+r_c1 )

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3 +r_c2)

        distilled_c4 = self.act(self.c4_d(r_c3))
        r_c4 = (self.c1_r(r_c3))
        r_c4 = self.act(r_c4+r_c3 )
        #
        distilled_c5 = self.act(self.c5_d(r_c4))
        r_c5 = (self.c2_r(r_c4))
        r_c5 = self.act(r_c5 +r_c4)
        #
        distilled_c6 = self.act(self.c6_d(r_c5))
        r_c6 = (self.c3_r(r_c5))
        r_c6 = self.act(r_c6 +r_c5 )
        #
        # distilled_c7 = self.act(self.c7_d(r_c6))
        # r_c7 = (self.c1_r(r_c6))
        # r_c7 = self.act(r_c7 + r_c6)
        # #
        # distilled_c8 = self.act(self.c8_d(r_c7))
        # r_c8 = (self.c2_r(r_c7))
        # r_c8 = self.act(r_c8 + r_c7)
        # #
        # distilled_c9 = self.act(self.c9_d(r_c8))
        # r_c9 = (self.c3_r(r_c8))
        # r_c9 = self.act(r_c9 + r_c8)

        r_c6 = self.act(self.c4(r_c6))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3,#distilled_c4,distilled_c5,
                         distilled_c4, distilled_c5, distilled_c6,
                         #distilled_c7, distilled_c8, distilled_c9,
                         r_c6], dim=1)
        out = self.c5(out)
        out_fused = self.esa(out)
        out_fused = self.cca(out_fused)
        return out_fused + input


class HLFA(nn.Module):

    def __init__(self, nc,scale,alpha,ratio=0.25):
        super(HLFA, self).__init__()
        self.channel = nc
        self.scale = scale
        self.alpha = alpha
        self.begin = nn.Sequential(
            nn.Conv2d(63, nc, 3, 1, 1),
            nn.GELU(),
        )
        self.feature = Susi0(nc)

        self.v1 = nn.Conv2d(nc, nc, 1)
        self.v2 = nn.Conv2d(nc, nc, 1)
        self.v3 = nn.Conv2d(nc, nc, 1)

        self.c2 = nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=1)
        self.merge = nn.Conv2d(nc * 3, nc, 1)
        self.esa = ESA(nc, Baseblock_, ratio)
        self.cca = CCALayer(nc)

        self.upconv1 = nn.Conv2d(nc, nc//2, 3, 1, 1, bias=True)
        self.HRconv1 = nn.Conv2d(nc//2, nc//2, 3, 1, 1, bias=True)
        # self.block1 = BSConvU(nc//2, nc//8)

        if self.scale == 4:
            self.upconv2 = nn.Conv2d(nc//2, nc//2, 3, 1, 1, bias=True)
            self.HRconv2 = nn.Conv2d(nc//2, nc//2, 3, 1, 1, bias=True)
            # self.block2 = BSConvU(nc//4, nc//16)

        self.conv_last = nn.Conv2d(nc//2, 3, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        _, c, _, _ = x.shape
        x_ = []
        for i in range(math.ceil(self.channel // c)):
            x_.append(x)
        x_ = torch.cat(x_, dim=1)
        x_ = self.begin(x_)
        fea1 = self.feature(x_)
        fea1 = self.v1(fea1)
        fea2 = self.feature(fea1)
        fea2 = self.v2(fea2)
        fea3 = self.feature(fea2)
        fea3 = self.v3(fea3)
        out = self.merge(torch.cat([fea3, fea2, fea1], dim=1))
        out = self.esa(out)
        out = self.cca(out)

        if self.scale == 2 or self.scale == 3:
            fea = self.upconv1(F.interpolate(out, scale_factor=self.scale, mode='nearest'))
            fea = self.lrelu(self.HRconv1(fea))
            # fea = self.block1(fea)
        elif self.scale == 4:
            fea = self.upconv1(F.interpolate(out, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.HRconv1(fea))
            # fea = self.block1(fea)
            fea = self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.HRconv2(fea))
            # fea = self.block2(fea)

        out = self.conv_last(fea)

        ILR = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        out = out + ILR
        return out