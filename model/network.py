import torch
from torch import nn
import torch.nn.functional as F


class BSConvU(torch.nn.Module):
    def __init__(self, in_channels, f, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()

        # pointwise
        self.pw = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # depthwise
        self.dw = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea


# 基础模块
class Baseblock_(nn.Module):
    def __init__(self,nc,f, kernel_size=1, stride =1):
        super(Baseblock_, self).__init__()
        padding = 1 if kernel_size==3 else 0
        self.feature = nn.Sequential(
            nn.Conv2d(nc, f, kernel_size,padding=padding,stride=stride),
            nn.GELU()
        )

    def forward(self, x):
        out = self.feature(x)
        return out

class Baseblock(nn.Module):
    def __init__(self,nc,f, kernel_size=1, stride =1):
        super(Baseblock, self).__init__()
        padding = 1 if kernel_size==3 else 0
        self.feature = nn.Sequential(
            nn.Conv2d(nc, f, kernel_size,padding=padding,stride=stride,groups=f),
            nn.Conv2d(f,f,1),
            nn.Conv2d(f, nc, kernel_size,padding=padding,stride=stride,groups=f),
            nn.Conv2d(nc, nc, 1),
            nn.GELU()
        )

    def forward(self, x):
        out = self.feature(x)
        return out

# 注意力模块
def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

class CCALayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(CCALayer, self).__init__()
        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class ESA(nn.Module):
    def __init__(self, n_feats, conv, ratio):
        super(ESA, self).__init__()
        f = int(n_feats * ratio)
        self.conv1 = conv(n_feats, f,  kernel_size=1 )
        self.conv_f = conv(f,  f,  kernel_size=1)
        self.conv_max = conv(f, f,   kernel_size=3)
        self.conv2 = conv(f, f, kernel_size=3, stride=2)
        self.conv3 = conv(f, f,   kernel_size=3 )
        self.conv3_ = conv(f, f,  kernel_size=3)
        self.conv4 = conv(f, n_feats,   kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m

# 上采样模块
class PA(nn.Module):
    '''PA is pixel attention'''

    def __init__(self, nf):
        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out


class NearestConv(nn.Module):
    def __init__(self, in_ch, num_feat, num_out_ch, scale= 4, conv=nn.Conv2d):
        super(NearestConv, self).__init__()
        self.scale = scale
        self.conv_before_upsample = nn.Sequential(
            conv(in_ch, num_feat, 3,1,1),
            nn.LeakyReLU(inplace=True),
        )

        self.conv_up1 = conv(num_feat, num_feat//2, 3, 1, 1)
        if self.scale == 4:
            self.conv_up2 = conv(num_feat//2, num_feat//4, 3, 1, 1)

        self.conv_hr = conv(num_feat//4, num_feat//4, 3, 1, 1)
        self.conv_last = conv(num_feat//4, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv_before_upsample(x)
        x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='bicubic')))
        if self.scale == 4:
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='bicubic')))
        x = self.conv_last(self.lrelu(self.conv_hr(x)))
        return x

class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


class PixelShuffleDirect(nn.Module):
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        super(PixelShuffleDirect, self).__init__()
        self.upsampleOneStep = UpsampleOneStep(scale, num_feat, num_out_ch, input_resolution=None)

    def forward(self, x):
        return self.upsampleOneStep(x)

class Fluctuation(nn.Module):
    def __init__(self, nc):
        super(Fluctuation, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(nc,nc,1),
            nn.GELU(),
        )

    def forward(self,x):
        return self.feature(x)


