import torch.nn as nn
import torch.nn.init as init
from math import floor
import torch.nn.functional as F
import torch

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class DnCNN(nn.Module):
    def __init__(self, depth=22, n_channels=64, image_channels=3, use_bnorm=True, kernel_size=3, find_noise=True, relu='relu'):
        super(DnCNN, self).__init__()
        padding = int((kernel_size-1)/2)
        layers = []
        self.find_noise = find_noise

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels,
                      kernel_size=kernel_size, padding=padding, bias=True))
        if relu == 'relu':
            layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.RReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels,
                          kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            if relu == 'relu':
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.RReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels,
                      kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        if self.find_noise:
            y = x
            out = self.dncnn(x)
            return y-out
        else:
            out = self.dncnn(x)
            return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

# based on https://arxiv.org/pdf/1603.05027.pdf ish
class RedishCNN(nn.Module):
    def __init__(self, n_channels=128, image_channels=3, kernel_size=5, depth=30, find_noise = False):
        super(RedishCNN, self).__init__()
        self.depth = depth
        self.bn = nn.BatchNorm2d(n_channels)
        self.conv_first = nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, stride=1, padding=0)
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, stride=1, padding=0)
        self.deconv = nn.ConvTranspose2d(n_channels, n_channels, kernel_size=kernel_size, stride=1, padding=0)
        self.deconv_last = nn.ConvTranspose2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, stride=1, padding=0)
        self.relu = nn.RReLU()
        self.find_noise = find_noise

    def forward(self, x):
        residuals = []
        layer = self.relu(self.conv_first(x))   #c1
        residuals.append(layer.clone())
        for _ in range(int(floor(self.depth-6)/2)):
            for _ in range(2):
                layer = self.bn(layer)
                layer = self.relu(layer)
                layer = self.conv(layer)
            residuals.append(layer.clone())
        layer = self.relu(self.conv(layer))     #clast
        layer = self.relu(self.deconv(layer))   #d1
        layer = self.relu(layer+residuals.pop())
        for _ in range(int(floor(self.depth-6)/2)):
            for _ in range(2):
                layer = self.bn(layer)
                layer = self.relu(layer)
                layer = self.deconv(layer)
            layer = self.relu(layer+residuals.pop())
        layer = self.relu(self.deconv_last(layer))
        if self.find_noise:
            return x - layer
        else:
            return layer

class RedCNN(nn.Module):
    def __init__(self, n_channels=128, image_channels=3, kernel_size=5, depth=30, relu='relu', find_noise = False):
        super(RedCNN, self).__init__()
        self.depth = depth
        self.conv_first = nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, stride=1, padding=0)
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, stride=1, padding=0)
        self.deconv = nn.ConvTranspose2d(n_channels, n_channels, kernel_size=kernel_size, stride=1, padding=0)
        self.deconv_last = nn.ConvTranspose2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, stride=1, padding=0)
        if relu == 'relu':
            self.relu = nn.ReLU(inplace=True)
        else:   # RReLU
            self.relu = nn.RReLU(inplace=True)
        #self.find_noise = find_noise

    def forward(self, x):
        residuals = []
        layer = self.relu(self.conv_first(x))   #c1
        layer = self.relu(self.conv(layer))     #c2
        residuals.append(layer.clone())
        for _ in range(int(floor(self.depth-6)/2)):
            layer = self.relu(self.conv(layer))
            layer = self.relu(self.conv(layer))
            residuals.append(layer.clone())
        layer = self.relu(self.conv(layer))     #clast
        layer = self.relu(self.deconv(layer))   #d1
        layer = self.relu(layer+residuals.pop())
        for _ in range(int(floor(self.depth-6)/2)):
            layer = self.relu(self.deconv(layer))
            layer = self.relu(self.deconv(layer))
            layer = self.relu(layer+residuals.pop())
        layer = self.relu(self.deconv(layer))
        layer = self.relu(self.deconv_last(layer))
        #if self.find_noise:
        #    return x - layer
        #else:
        #    return layer
        return layer


# UNet with upconv


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, find_noise=False):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.find_noise = find_noise

    def forward(self, x):
        if self.find_noise:
            y = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        if self.find_noise:
            return y - F.sigmoid(x)
        return F.sigmoid(x)

class HunkyNet(nn.Module):
    # possible input size: 224+int*16
    def __init__(self):
        super(HunkyNet, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.down = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 96, 3),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        # down
        self.enc3 = nn.Sequential(
            nn.Conv2d(96,128,3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # down
        self.enc4 = nn.Sequential(
            nn.Conv2d(128,256,3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # down
        self.enc5 = nn.Sequential(
        nn.Conv2d(256,512,3),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Conv2d(512,512,3),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        )
        # down
        self.encdec = nn.Sequential(
            nn.Conv2d(512,1024,3),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024,1024,3),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.ConvTranspose2d(1024,512,2,stride=2)
        self.dec2 = nn.Sequential(
            # += clone
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512,512,3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512,512,3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(512,256,2,stride=2)
        self.dec3 = nn.Sequential(
            # += clone
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256,256,3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256,256,3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.ConvTranspose2d(256,128,2,stride=2)
        self.dec4 = nn.Sequential(
            # += clone
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128,128,3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128,128,3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.up4 = nn.ConvTranspose2d(128,96,2,stride=2)
        self.dec5 = nn.Sequential(
            # += clone
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96,96,3),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96,96,3),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        self.up5 = nn.ConvTranspose2d(96,64,2,stride=2)
        self.dec6 = nn.Sequential(
            # += clone
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,64,5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,3,5),
            nn.Sigmoid(),
        )
    def forward(self, x):
        residuals = []
        layer = self.enc1(x)
        residuals.append(layer.clone())
        layer = self.down(layer)
        layer = self.enc2(layer)
        residuals.append(layer.clone())
        layer = self.down(layer)
        layer = self.enc3(layer)
        residuals.append(layer.clone())
        layer = self.down(layer)
        layer = self.enc4(layer)
        residuals.append(layer.clone())
        layer = self.down(layer)
        layer = self.enc5(layer)
        residuals.append(layer.clone())
        layer = self.down(layer)
        layer = self.encdec(layer)
        layer = self.up1(layer)
        layer = layer+residuals.pop()
        layer = self.dec2(layer)
        layer = self.up2(layer)
        layer = layer+residuals.pop()
        layer = self.dec3(layer)
        layer = self.up3(layer)
        layer = layer+residuals.pop()
        layer = self.dec4(layer)
        layer = self.up4(layer)
        layer = layer+residuals.pop()
        layer = self.dec5(layer)
        layer = self.up5(layer)
        layer = layer+residuals.pop()
        layer = self.dec6(layer)
        return layer

# 256-px
class HunkyDisc(nn.Module):
    def __init__(self, input_channels):
        super(HunkyDisc, self).__init__()
        #256
        self.enc = nn.Sequential(
            nn.Conv2d(input_channels, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 248
            nn.MaxPool2d(2),
            # 124
            nn.Conv2d(64, 96, 3),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            # 120
            nn.MaxPool2d(2),
            nn.Conv2d(96,128,3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 56
            nn.MaxPool2d(2),
            nn.Conv2d(128,256,3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 24
            nn.MaxPool2d(2),
            nn.Conv2d(256,512,3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 8
            nn.MaxPool2d(2),
            nn.Conv2d(512,1024,3),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1, 2),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.enc(x)

class HunNet(nn.Module):
    def __init__(self):
        funit = 32
        super(HunNet, self).__init__()
        self.enc160to158std = nn.Sequential(
            nn.Conv2d(3, 4*funit, 3),
            #nn.BatchNorm2d(4*funit),
            nn.ReLU(inplace=True),
        )
        self.enc158to154std = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, bias=False),
            nn.BatchNorm2d(3*funit),
            nn.ReLU(inplace=True),
            nn.Conv2d(3*funit, 3*funit, 3, bias=False),
            nn.BatchNorm2d(3*funit),
            nn.ReLU(inplace=True),
        )
        self.enc154to150std = nn.Sequential(
            nn.Conv2d(6*funit, 4*funit, 3, bias=False),
            nn.BatchNorm2d(4*funit),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*funit, 4*funit, 3, bias=False),
            nn.BatchNorm2d(4*funit),
            nn.ReLU(inplace=True),
        )
        self.enc158to154dil = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(3*funit),
            nn.ReLU(inplace=True),
        )
        self.enc154to150dil = nn.Sequential(
            nn.Conv2d(6*funit, 4*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(4*funit),
            nn.ReLU(inplace=True),
        )
        self.enc160to150dil = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3, dilation=5, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc150to50str = nn.Sequential(
            nn.Conv2d(10*funit, 10*funit, 3, stride=3, bias=False),
            nn.BatchNorm2d(10*funit),
            nn.ReLU(inplace=True),
        )
        self.enc50to46std = nn.Sequential(
            nn.Conv2d(10*funit, 5*funit, 3, bias=False),
            nn.BatchNorm2d(5*funit),
            nn.ReLU(inplace=True),
            nn.Conv2d(5*funit, 5*funit, 3, bias=False),
            nn.BatchNorm2d(5*funit),
            nn.ReLU(inplace=True),
        )
        self.enc46to42std = nn.Sequential(
            nn.Conv2d(10*funit, 8*funit, 3, bias=False),
            nn.BatchNorm2d(8*funit),
            nn.ReLU(inplace=True),
            nn.Conv2d(8*funit, 8*funit, 3, bias=False),
            nn.BatchNorm2d(8*funit),
            nn.ReLU(inplace=True),
        )
        self.enc50to46dil = nn.Sequential(
            nn.Conv2d(10*funit, 5*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(5*funit),
            nn.ReLU(inplace=True),
        )
        self.enc46to42dil = nn.Sequential(
            nn.Conv2d(10*funit, 8*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(8*funit),
            nn.ReLU(inplace=True),
        )
        self.enc42to14str = nn.Sequential(
            nn.Conv2d(16*funit, 16*funit, 3, stride=3, bias=False),
            nn.BatchNorm2d(16*funit),
            nn.ReLU(inplace=True),
        )

        self.enc14to10std = nn.Sequential(
            nn.Conv2d(16*funit, 8*funit, 3, bias=False),
            nn.BatchNorm2d(8*funit),
            nn.ReLU(inplace=True),
            nn.Conv2d(8*funit, 8*funit, 3, bias=False),
            nn.BatchNorm2d(8*funit),
            nn.ReLU(inplace=True),
        )
        self.enc10to6std = nn.Sequential(
            nn.Conv2d(16*funit, 16*funit, 3, bias=False),
            nn.BatchNorm2d(16*funit),
            nn.ReLU(inplace=True),
            nn.Conv2d(16*funit, 16*funit, 3, bias=False),
            nn.BatchNorm2d(16*funit),
            nn.ReLU(inplace=True),
        )
        self.enc14to10dil = nn.Sequential(
            nn.Conv2d(16*funit, 8*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(8*funit),
            nn.ReLU(inplace=True),
        )
        self.enc10to6dil = nn.Sequential(
            nn.Conv2d(16*funit, 16*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(16*funit),
            nn.ReLU(inplace=True),
        )
        self.enc6to3str = nn.Sequential(
            nn.Conv2d(32*funit, 32*funit, 2, stride=2, bias=False),
            nn.BatchNorm2d(32*funit),
            nn.ReLU(inplace=True),
        )
        self.enc3to1std = nn.Sequential(
            nn.Conv2d(32*funit, 32*funit, 3, bias=False),
            nn.BatchNorm2d(32*funit),
            nn.ReLU(inplace=True),
        )
        self.dec1to3std = nn.Sequential(
            # in: 32
            # out: 32
            nn.ConvTranspose2d(32*funit, 32*funit, 3, bias=False),
            nn.BatchNorm2d(32*funit),
            nn.ReLU(inplace=True),
        )
        self.dec3to6str = nn.Sequential(
            # in: 32+32
            # out: 32
            nn.ConvTranspose2d(64*funit, 32*funit, 2, stride=2, bias=False),
            nn.BatchNorm2d(32*funit),
            nn.ReLU(inplace=True),
        )

        self.dec6to10std = nn.Sequential(
            # in: 32+32, out: 8
            nn.ConvTranspose2d(64*funit, 8*funit, 3, bias=False),
            nn.BatchNorm2d(8*funit),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8*funit, 8*funit, 3, bias=False),
            nn.BatchNorm2d(8*funit),
            nn.ReLU(inplace=True),
        )

        self.dec6to10dil = nn.Sequential(
            # in: 32+32, out: 8
            nn.ConvTranspose2d(64*funit, 8*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(8*funit),
            nn.ReLU(inplace=True),
        )
        self.dec10to14std = nn.Sequential(
            # in: 16+16, out: 8
            nn.ConvTranspose2d(32*funit, 8*funit, 3, bias=False),
            nn.BatchNorm2d(8*funit),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8*funit, 8*funit, 3, bias=False),
            nn.BatchNorm2d(8*funit),
            nn.ReLU(inplace=True),

        )
        self.dec10to14dil = nn.Sequential(
            # in: 16+16, out:8
            nn.ConvTranspose2d(32*funit, 8*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(8*funit),
            nn.ReLU(inplace=True),
        )
        self.dec14to42str = nn.Sequential(
            # in: 16+16, out:16
            nn.ConvTranspose2d(32*funit, 16*funit, 3, stride=3, bias=False),
            nn.BatchNorm2d(16*funit),
            nn.ReLU(inplace=True),
        )
        self.dec42to46std = nn.Sequential(
            # in: 16+16, out: 5
            nn.ConvTranspose2d(32*funit, 5*funit, 3, bias=False),
            nn.BatchNorm2d(5*funit),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(5*funit, 5*funit, 3, bias=False),
            nn.BatchNorm2d(5*funit),
            nn.ReLU(inplace=True),
        )
        self.dec42to46dil = nn.Sequential(
            # in: 16+16, out:5
            nn.ConvTranspose2d(32*funit, 5*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(5*funit),
            nn.ReLU(inplace=True),
        )
        self.dec46to50std = nn.Sequential(
            # in: 10+10, out: 5
            nn.ConvTranspose2d(20*funit, 5*funit, 3, bias=False),
            nn.BatchNorm2d(5*funit),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(5*funit, 5*funit, 3, bias=False),
            nn.BatchNorm2d(5*funit),
            nn.ReLU(inplace=True),
        )
        self.dec46to50dil = nn.Sequential(
            # in: 10+10, out: 5
            nn.ConvTranspose2d(20*funit, 5*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(5*funit),
            nn.ReLU(inplace=True),
        )
        self.dec50to150str = nn.Sequential(
            # in: 10+10, out: 10
            nn.ConvTranspose2d(20*funit, 10*funit, 3, stride=3, bias=False),
            nn.BatchNorm2d(10*funit),
            nn.ReLU(inplace=True),
        )
        self.dec150to154std = nn.Sequential(
            # in: 10+10, out: 3
            nn.ConvTranspose2d(20*funit, 3*funit, 3, bias=False),
            nn.BatchNorm2d(3*funit),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            nn.BatchNorm2d(3*funit),
            nn.ReLU(inplace=True),
        )
        self.dec150to154dil = nn.Sequential(
            # in: 10+10, out: 3
            nn.ConvTranspose2d(20*funit, 3*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(3*funit),
            nn.ReLU(inplace=True),
        )
        self.dec154to158std = nn.Sequential(
            # in: 6+6, out: 2
            nn.ConvTranspose2d(12*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.dec154to158dil = nn.Sequential(
            # in: 6+6, out: 2
            nn.ConvTranspose2d(12*funit, 2*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.dec158to160std = nn.Sequential(
            # in: 4+4
            nn.ConvTranspose2d(8*funit, 3, 3),
            nn.Sigmoid(),
        )


    def forward(self, x):
        # 160 to 150
        dil160 = x.clone()
        std158 = self.enc160to158std(x)
        del(x)
        dil158 = std158.clone()
        upc158 = std158.clone()
        std154 = self.enc158to154std(std158)
        del(std158)
        dil154 = self.enc158to154dil(dil158)
        del(dil158)
        cat154 = torch.cat([std154, dil154], 1)
        del(std154)
        dil154 = cat154.clone()
        upc154 = cat154.clone()
        std150 = self.enc154to150std(cat154)
        del(cat154)
        dil150 = self.enc154to150dil(dil154)
        del(dil154)
        dil3_150 = self.enc160to150dil(dil160)
        del(dil160)

        cat150 = torch.cat([std150, dil150, dil3_150], 1)
        del(std150, dil150, dil3_150)
        upc150 = cat150.clone()
        str50 = self.enc150to50str(cat150)
        del(cat150)
        dil50 = str50.clone()
        upc50 = str50.clone()
        std46 = self.enc50to46std(str50)
        del(str50)
        dil46 = self.enc50to46dil(dil50)
        del(dil50)
        cat46 = torch.cat([std46, dil46], 1)
        del(std46)
        dil46 = cat46.clone()

        upc46 = cat46.clone()
        std42 = self.enc46to42std(cat46)
        del(cat46)
        dil42 = self.enc46to42dil(dil46)
        del(dil46)

        cat42 = torch.cat([std42, dil42], 1)
        del(std42, dil42)
        upc42 = cat42.clone()
        str14 = self.enc42to14str(cat42)
        dil14 = str14.clone()
        upc14 = str14.clone()
        cat10 = torch.cat([self.enc14to10std(str14), self.enc14to10dil(dil14)], 1)
        del(str14, dil14)
        dil10 = cat10.clone()
        upc10 = cat10.clone()
        cat6 = torch.cat([self.enc10to6std(cat10), self.enc10to6dil(dil10)], 1)
        del(cat10, dil10)
        upc6 = cat6.clone()

        str3 = self.enc6to3str(cat6)    # k2s2
        del(cat6)
        upc3 = str3.clone()
        std1 = self.enc3to1std(str3)
        del(str3)

        # up

        std3 = torch.cat([upc3, self.dec1to3std(std1)], 1)
        del(std1)

        str6 = torch.cat([upc6, self.dec3to6str(std3)], 1)
        del(upc6, std3)
        dil6 = str6.clone()
        cat10 = torch.cat([upc10, self.dec6to10std(str6), self.dec6to10dil(dil6)], 1)
        del(dil6, str6, upc10)
        dil10 = cat10.clone()
        cat14 = torch.cat([upc14, self.dec10to14std(cat10), self.dec10to14dil(dil10)], 1)
        del(cat10, dil10, upc14)
        cat42 = torch.cat([upc42, self.dec14to42str(cat14)], 1)
        del(cat14, upc42)
        dil42 = cat42.clone()
        cat46 = torch.cat([upc46, self.dec42to46std(cat42), self.dec42to46dil(dil42)], 1)
        del(dil42, cat42, upc46)
        dil46 = cat46.clone()
        cat50 = torch.cat([upc50, self.dec46to50std(cat46), self.dec46to50dil(dil46)], 1)
        del(cat46, dil46, upc50)
        cat150 = torch.cat([upc150, self.dec50to150str(cat50)], 1)
        del(upc150, cat50)
        dil150 = cat150.clone()
        cat154 = torch.cat([upc154, self.dec150to154std(cat150), self.dec150to154dil(dil150)], 1)
        del(cat150, dil150, upc154)
        dil154 = cat154.clone()
        cat158 = torch.cat([upc158, self.dec154to158std(cat154), self.dec154to158dil(dil154)], 1)
        del(dil154, upc158, cat154)
        return self.dec158to160std(cat158)

#160
class HuNet(nn.Module):
    def __init__(self):
        funit = 32
        super(HuNet, self).__init__()
        self.enc160to158std = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3),
            #nn.BatchNorm2d(4*funit),
            nn.PReLU(),
        )
        self.enc158to154std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc154to150std = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, bias=False),
            #nn.BatchNorm2d(3*funit),
            nn.PReLU(),
            nn.Conv2d(3*funit, 3*funit, 3, bias=False),
            #nn.BatchNorm2d(3*funit),
            nn.PReLU(),
        )
        self.enc158to154dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc154to150dil = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(3*funit),
            nn.PReLU(),
        )
        self.enc160to150dil = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3, dilation=5, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc150to50str = nn.Sequential(
            nn.Conv2d(8*funit, 2*funit, 3, stride=3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc50to46std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc46to42std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc50to46dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc46to42dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc42to14str = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, stride=3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )

        self.enc14to10std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc10to6std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc14to10dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc10to6dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc6to3str = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 2, stride=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc3to1std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec1to3std = nn.Sequential(
            # in: 2
            # out: 2
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec3to6str = nn.Sequential(
            # in: 2+2
            # out: 2
            nn.ConvTranspose2d(4*funit, 2*funit, 2, stride=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )

        self.dec6to10std = nn.Sequential(
            # in: 2+4, out: 2
            nn.ConvTranspose2d(6*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )

        self.dec6to10dil = nn.Sequential(
            # in: 2+4, out: 2
            nn.ConvTranspose2d(6*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec10to14std = nn.Sequential(
            # in: 4+4, out: 2
            nn.ConvTranspose2d(8*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),

        )
        self.dec10to14dil = nn.Sequential(
            # in: 4+4, out:2
            nn.ConvTranspose2d(8*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec14to42str = nn.Sequential(
            # in: 4+2, out:2
            nn.ConvTranspose2d(6*funit, 2*funit, 3, stride=3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec42to46std = nn.Sequential(
            # in: 2+4, out: 2
            nn.ConvTranspose2d(6*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec42to46dil = nn.Sequential(
            # in: 2+4, out:2
            nn.ConvTranspose2d(6*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec46to50std = nn.Sequential(
            # in: 4+4, out: 2
            nn.ConvTranspose2d(8*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec46to50dil = nn.Sequential(
            # in: 4+4, out: 2
            nn.ConvTranspose2d(8*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec50to150str = nn.Sequential(
            # in: 4+2, out: 4
            nn.ConvTranspose2d(6*funit, 4*funit, 3, stride=3, bias=False),
            #nn.BatchNorm2d(4*funit),
            nn.PReLU(),
        )
        self.dec150to154std = nn.Sequential(
            # in: 8+4, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, bias=False),
            #nn.BatchNorm2d(3*funit),
            nn.PReLU(),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            #nn.BatchNorm2d(3*funit),
            nn.PReLU(),
        )
        self.dec150to154dil = nn.Sequential(
            # in: 8+4, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(3*funit),
            nn.PReLU(),
        )
        self.dec154to158std = nn.Sequential(
            # in: 6+4, out: 2
            nn.ConvTranspose2d(10*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec154to158dil = nn.Sequential(
            # in: 6+4, out: 2
            nn.ConvTranspose2d(10*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec158to160std = nn.Sequential(
            # in: 4+2
            nn.ConvTranspose2d(6*funit, 3, 3),
            nn.ReLU(),
        )


    def forward(self, x):
        # 160 to 150
        dil160 = x.clone()
        std158 = self.enc160to158std(x)
        del(x)
        dil158 = std158.clone()
        upc158 = std158.clone()
        std154 = self.enc158to154std(std158)
        del(std158)
        dil154 = self.enc158to154dil(dil158)
        del(dil158)
        cat154 = torch.cat([std154, dil154], 1)
        del(std154)
        dil154 = cat154.clone()
        upc154 = cat154.clone()
        std150 = self.enc154to150std(cat154)
        del(cat154)
        dil150 = self.enc154to150dil(dil154)
        del(dil154)
        dil3_150 = self.enc160to150dil(dil160)
        del(dil160)

        cat150 = torch.cat([std150, dil150, dil3_150], 1)
        del(std150, dil150, dil3_150)
        upc150 = cat150.clone()
        str50 = self.enc150to50str(cat150)
        del(cat150)
        dil50 = str50.clone()
        upc50 = str50.clone()
        std46 = self.enc50to46std(str50)
        del(str50)
        dil46 = self.enc50to46dil(dil50)
        del(dil50)
        cat46 = torch.cat([std46, dil46], 1)
        del(std46)
        dil46 = cat46.clone()

        upc46 = cat46.clone()
        std42 = self.enc46to42std(cat46)
        del(cat46)
        dil42 = self.enc46to42dil(dil46)
        del(dil46)

        cat42 = torch.cat([std42, dil42], 1)
        del(std42, dil42)
        upc42 = cat42.clone()
        str14 = self.enc42to14str(cat42)
        dil14 = str14.clone()
        upc14 = str14.clone()
        cat10 = torch.cat([self.enc14to10std(str14), self.enc14to10dil(dil14)], 1)
        del(str14, dil14)
        dil10 = cat10.clone()
        upc10 = cat10.clone()
        cat6 = torch.cat([self.enc10to6std(cat10), self.enc10to6dil(dil10)], 1)
        del(cat10, dil10)
        upc6 = cat6.clone()

        str3 = self.enc6to3str(cat6)    # k2s2
        del(cat6)
        upc3 = str3.clone()
        std1 = self.enc3to1std(str3)
        del(str3)

        # up

        std3 = torch.cat([upc3, self.dec1to3std(std1)], 1)
        del(std1)

        str6 = torch.cat([upc6, self.dec3to6str(std3)], 1)
        del(upc6, std3)
        dil6 = str6.clone()
        cat10 = torch.cat([upc10, self.dec6to10std(str6), self.dec6to10dil(dil6)], 1)
        del(dil6, str6, upc10)
        dil10 = cat10.clone()
        cat14 = torch.cat([upc14, self.dec10to14std(cat10), self.dec10to14dil(dil10)], 1)
        del(cat10, dil10, upc14)
        cat42 = torch.cat([upc42, self.dec14to42str(cat14)], 1)
        del(cat14, upc42)
        dil42 = cat42.clone()
        cat46 = torch.cat([upc46, self.dec42to46std(cat42), self.dec42to46dil(dil42)], 1)
        del(dil42, cat42, upc46)
        dil46 = cat46.clone()
        cat50 = torch.cat([upc50, self.dec46to50std(cat46), self.dec46to50dil(dil46)], 1)
        del(cat46, dil46, upc50)
        cat150 = torch.cat([upc150, self.dec50to150str(cat50)], 1)
        del(upc150, cat50)
        dil150 = cat150.clone()
        cat154 = torch.cat([upc154, self.dec150to154std(cat150), self.dec150to154dil(dil150)], 1)
        del(cat150, dil150, upc154)
        dil154 = cat154.clone()
        cat158 = torch.cat([upc158, self.dec154to158std(cat154), self.dec154to158dil(dil154)], 1)
        del(dil154, upc158, cat154)
        return self.dec158to160std(cat158)

#160
class HulNet(nn.Module):
    def __init__(self):
        funit = 32
        super(HulNet, self).__init__()
        self.enc160to158std = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3),
            #nn.BatchNorm2d(4*funit),
            nn.PReLU(),
        )
        self.enc158to154std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc154to150std = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, bias=False),
            #nn.BatchNorm2d(3*funit),
            nn.PReLU(),
            nn.Conv2d(3*funit, 3*funit, 3, bias=False),
            #nn.BatchNorm2d(3*funit),
            nn.PReLU(),
        )
        self.enc158to154dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc154to150dil = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(3*funit),
            nn.PReLU(),
        )
        self.enc160to150dil = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3, dilation=5, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc150to50str = nn.Sequential(
            nn.Conv2d(8*funit, 2*funit, 3, stride=3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc50to46std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc46to42std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc50to46dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc46to42dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc42to14str = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, stride=3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )

        self.enc14to10std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc10to6std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc14to10dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc10to6dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc6to3str = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 2, stride=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.enc3to1std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec1to3std = nn.Sequential(
            # in: 2
            # out: 2
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec3to6str = nn.Sequential(
            # in: 2+2
            # out: 2
            nn.ConvTranspose2d(4*funit, 2*funit, 2, stride=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )

        self.dec6to10std = nn.Sequential(
            # in: 2+4, out: 2
            nn.ConvTranspose2d(6*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )

        self.dec6to10dil = nn.Sequential(
            # in: 2+4, out: 2
            nn.ConvTranspose2d(6*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec10to14std = nn.Sequential(
            # in: 4+4, out: 2
            nn.ConvTranspose2d(8*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),

        )
        self.dec10to14dil = nn.Sequential(
            # in: 4+4, out:2
            nn.ConvTranspose2d(8*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec14to42str = nn.Sequential(
            # in: 4+2, out:2
            nn.ConvTranspose2d(6*funit, 2*funit, 3, stride=3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec42to46std = nn.Sequential(
            # in: 2+4, out: 2
            nn.ConvTranspose2d(6*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec42to46dil = nn.Sequential(
            # in: 2+4, out:2
            nn.ConvTranspose2d(6*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec46to50std = nn.Sequential(
            # in: 4+4, out: 2
            nn.ConvTranspose2d(8*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec46to50dil = nn.Sequential(
            # in: 4+4, out: 2
            nn.ConvTranspose2d(8*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec50to150str = nn.Sequential(
            # in: 4+2, out: 4
            nn.ConvTranspose2d(6*funit, 4*funit, 3, stride=3, bias=False),
            #nn.BatchNorm2d(4*funit),
            nn.PReLU(),
        )
        self.dec150to154std = nn.Sequential(
            # in: 8+4, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, bias=False),
            #nn.BatchNorm2d(3*funit),
            nn.PReLU(),
            nn.ConvTranspose2d(3*funit, 3*funit, 3, bias=False),
            #nn.BatchNorm2d(3*funit),
            nn.PReLU(),
        )
        self.dec150to154dil = nn.Sequential(
            # in: 8+4, out: 3
            nn.ConvTranspose2d(12*funit, 3*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(3*funit),
            nn.PReLU(),
        )
        self.dec154to158std = nn.Sequential(
            # in: 6+4, out: 2
            nn.ConvTranspose2d(10*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
            nn.ConvTranspose2d(2*funit, 2*funit, 3, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec154to158dil = nn.Sequential(
            # in: 6+4, out: 2
            nn.ConvTranspose2d(10*funit, 2*funit, 3, dilation=2, bias=False),
            #nn.BatchNorm2d(2*funit),
            nn.PReLU(),
        )
        self.dec158to160std = nn.Sequential(
            # in: 4+2
            nn.ConvTranspose2d(6*funit, 3, 3),
            nn.ReLU(),
        )


    def forward(self, x):
        # 160 to 150
        dil160 = x#.clone()
        std158 = self.enc160to158std(x)
        del(x)
        dil158 = std158#.clone()
        upc158 = std158#.clone()
        std154 = self.enc158to154std(std158)
        del(std158)
        dil154 = self.enc158to154dil(dil158)
        del(dil158)
        cat154 = torch.cat([std154, dil154], 1)
        del(std154)
        dil154 = cat154#.clone()
        upc154 = cat154#.clone()
        std150 = self.enc154to150std(cat154)
        del(cat154)
        dil150 = self.enc154to150dil(dil154)
        del(dil154)
        dil3_150 = self.enc160to150dil(dil160)
        del(dil160)

        cat150 = torch.cat([std150, dil150, dil3_150], 1)
        del(std150, dil150, dil3_150)
        upc150 = cat150#.clone()
        str50 = self.enc150to50str(cat150)
        del(cat150)
        dil50 = str50#.clone()
        upc50 = str50#.clone()
        std46 = self.enc50to46std(str50)
        del(str50)
        dil46 = self.enc50to46dil(dil50)
        del(dil50)
        cat46 = torch.cat([std46, dil46], 1)
        del(std46)
        dil46 = cat46#.clone()

        upc46 = cat46#.clone()
        std42 = self.enc46to42std(cat46)
        del(cat46)
        dil42 = self.enc46to42dil(dil46)
        del(dil46)

        cat42 = torch.cat([std42, dil42], 1)
        del(std42, dil42)
        upc42 = cat42#.clone()
        str14 = self.enc42to14str(cat42)
        dil14 = str14#.clone()
        upc14 = str14#.clone()
        cat10 = torch.cat([self.enc14to10std(str14), self.enc14to10dil(dil14)], 1)
        del(str14, dil14)
        dil10 = cat10#.clone()
        upc10 = cat10#.clone()
        cat6 = torch.cat([self.enc10to6std(cat10), self.enc10to6dil(dil10)], 1)
        del(cat10, dil10)
        upc6 = cat6#.clone()

        str3 = self.enc6to3str(cat6)    # k2s2
        del(cat6)
        upc3 = str3#.clone()
        std1 = self.enc3to1std(str3)
        del(str3)

        # up

        std3 = torch.cat([upc3, self.dec1to3std(std1)], 1)
        del(std1)

        str6 = torch.cat([upc6, self.dec3to6str(std3)], 1)
        del(upc6, std3)
        dil6 = str6#.clone()
        cat10 = torch.cat([upc10, self.dec6to10std(str6), self.dec6to10dil(dil6)], 1)
        del(dil6, str6, upc10)
        dil10 = cat10#.clone()
        cat14 = torch.cat([upc14, self.dec10to14std(cat10), self.dec10to14dil(dil10)], 1)
        del(cat10, dil10, upc14)
        cat42 = torch.cat([upc42, self.dec14to42str(cat14)], 1)
        del(cat14, upc42)
        dil42 = cat42#.clone()
        cat46 = torch.cat([upc46, self.dec42to46std(cat42), self.dec42to46dil(dil42)], 1)
        del(dil42, cat42, upc46)
        dil46 = cat46#.clone()
        cat50 = torch.cat([upc50, self.dec46to50std(cat46), self.dec46to50dil(dil46)], 1)
        del(cat46, dil46, upc50)
        cat150 = torch.cat([upc150, self.dec50to150str(cat50)], 1)
        del(upc150, cat50)
        dil150 = cat150#.clone()
        cat154 = torch.cat([upc154, self.dec150to154std(cat150), self.dec150to154dil(dil150)], 1)
        del(cat150, dil150, upc154)
        dil154 = cat154#.clone()
        cat158 = torch.cat([upc158, self.dec154to158std(cat154), self.dec154to158dil(dil154)], 1)
        del(dil154, upc158, cat154)
        return self.dec158to160std(cat158)


#160
class HuDisc(nn.Module):
    def __init__(self):
        funit = 32
        super(HuDisc, self).__init__()
        self.enc160to158std = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3),
            #nn.BatchNorm2d(4*funit),
            nn.ReLU(inplace=True),
        )
        self.enc158to154std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc154to150std = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, bias=False),
            nn.BatchNorm2d(3*funit),
            nn.ReLU(inplace=True),
            nn.Conv2d(3*funit, 3*funit, 3, bias=False),
            nn.BatchNorm2d(3*funit),
            nn.ReLU(inplace=True),
        )
        self.enc158to154dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc154to150dil = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(3*funit),
            nn.ReLU(inplace=True),
        )
        self.enc160to150dil = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3, dilation=5, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc150to50str = nn.Sequential(
            nn.Conv2d(8*funit, 2*funit, 3, stride=3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc50to46std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc46to42std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc50to46dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc46to42dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc42to14str = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, stride=3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )

        self.enc14to10std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc10to6std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc14to10dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc10to6dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc6to3str = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 2, stride=2, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc3to1std = nn.Sequential(
            nn.Conv2d(2*funit, 1, 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 160 to 150
        dil160 = x.clone()
        std158 = self.enc160to158std(x)
        del(x)
        dil158 = std158.clone()
        std154 = self.enc158to154std(std158)
        del(std158)
        dil154 = self.enc158to154dil(dil158)
        del(dil158)
        cat154 = torch.cat([std154, dil154], 1)
        del(std154)
        dil154 = cat154.clone()
        std150 = self.enc154to150std(cat154)
        del(cat154)
        dil150 = self.enc154to150dil(dil154)
        del(dil154)
        dil3_150 = self.enc160to150dil(dil160)
        del(dil160)

        cat150 = torch.cat([std150, dil150, dil3_150], 1)
        del(std150, dil150, dil3_150)
        str50 = self.enc150to50str(cat150)
        del(cat150)
        dil50 = str50.clone()
        std46 = self.enc50to46std(str50)
        del(str50)
        dil46 = self.enc50to46dil(dil50)
        del(dil50)
        cat46 = torch.cat([std46, dil46], 1)
        del(std46)
        dil46 = cat46.clone()

        std42 = self.enc46to42std(cat46)
        del(cat46)
        dil42 = self.enc46to42dil(dil46)
        del(dil46)

        cat42 = torch.cat([std42, dil42], 1)
        del(std42, dil42)
        str14 = self.enc42to14str(cat42)
        dil14 = str14.clone()
        cat10 = torch.cat([self.enc14to10std(str14), self.enc14to10dil(dil14)], 1)
        del(str14, dil14)
        dil10 = cat10.clone()
        cat6 = torch.cat([self.enc10to6std(cat10), self.enc10to6dil(dil10)], 1)
        del(cat10, dil10)

        str3 = self.enc6to3str(cat6)    # k2s2
        del(cat6)
        return self.enc3to1std(str3)


#144
class Hu144Disc(nn.Module):
    def __init__(self):
        funit = 32
        super(HuDisc, self).__init__()
        self.enc144to142std = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3),
            #nn.BatchNorm2d(4*funit),
            nn.ReLU(inplace=True),
        )
        self.enc142to138std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc138to134std = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, bias=False),
            nn.BatchNorm2d(3*funit),
            nn.ReLU(inplace=True),
            nn.Conv2d(3*funit, 3*funit, 3, bias=False),
            nn.BatchNorm2d(3*funit),
            nn.ReLU(inplace=True),
        )
        self.enc142to138dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc138to134dil = nn.Sequential(
            nn.Conv2d(4*funit, 3*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(3*funit),
            nn.ReLU(inplace=True),
        )
        self.enc144to134dil = nn.Sequential(
            nn.Conv2d(3, 2*funit, 3, dilation=5, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc134to44str = nn.Sequential(
            nn.Conv2d(8*funit, 2*funit, 3, stride=3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc46to42std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc42to38std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc46to42dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc42to38dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc42to14str = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, stride=3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )

        self.enc14to10std = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc10to6std = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*funit, 2*funit, 3, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc14to10dil = nn.Sequential(
            nn.Conv2d(2*funit, 2*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc10to6dil = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 3, dilation=2, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc6to3str = nn.Sequential(
            nn.Conv2d(4*funit, 2*funit, 2, stride=2, bias=False),
            nn.BatchNorm2d(2*funit),
            nn.ReLU(inplace=True),
        )
        self.enc3to1std = nn.Sequential(
            nn.Conv2d(2*funit, 1, 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 160 to 150
        dil160 = x.clone()
        std158 = self.enc160to158std(x)
        del(x)
        dil158 = std158.clone()
        std154 = self.enc158to154std(std158)
        del(std158)
        dil154 = self.enc158to154dil(dil158)
        del(dil158)
        cat154 = torch.cat([std154, dil154], 1)
        del(std154)
        dil154 = cat154.clone()
        std150 = self.enc154to150std(cat154)
        del(cat154)
        dil150 = self.enc154to150dil(dil154)
        del(dil154)
        dil3_150 = self.enc160to150dil(dil160)
        del(dil160)

        cat150 = torch.cat([std150, dil150, dil3_150], 1)
        del(std150, dil150, dil3_150)
        str50 = self.enc150to50str(cat150)
        del(cat150)
        dil50 = str50.clone()
        std46 = self.enc50to46std(str50)
        del(str50)
        dil46 = self.enc50to46dil(dil50)
        del(dil50)
        cat46 = torch.cat([std46, dil46], 1)
        del(std46)
        dil46 = cat46.clone()

        std42 = self.enc46to42std(cat46)
        del(cat46)
        dil42 = self.enc46to42dil(dil46)
        del(dil46)

        cat42 = torch.cat([std42, dil42], 1)
        del(std42, dil42)
        str14 = self.enc42to14str(cat42)
        dil14 = str14.clone()
        cat10 = torch.cat([self.enc14to10std(str14), self.enc14to10dil(dil14)], 1)
        del(str14, dil14)
        dil10 = cat10.clone()
        cat6 = torch.cat([self.enc10to6std(cat10), self.enc10to6dil(dil10)], 1)
        del(cat10, dil10)

        str3 = self.enc6to3str(cat6)    # k2s2
        del(cat6)
        return self.enc3to1std(str3)


