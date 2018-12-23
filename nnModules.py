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
    def __init__(self, depth=22, n_channels=64, image_channels=3, use_bnorm=True, kernel_size=3, find_noise=True):
        super(DnCNN, self).__init__()
        padding = int((kernel_size-1)/2)
        layers = []
        self.find_noise = find_noise

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels,
                      kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.RReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels,
                          kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
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

# because I can (not done yet)
# class DecEncCNN(nn.Module):
#     def __init__(self, n_channels=128, image_channels=3, kernel_size=5, depth=30):
#         super(DecEncCNN, self).__init__()
#         self.depth = depth
#         self.bn = nn.BatchNorm2d(n_channels)
#         self.conv_first = nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, stride=1, padding=0)
#         self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, stride=1, padding=0)
#         self.deconv = nn.ConvTranspose2d(n_channels, n_channels, kernel_size=kernel_size, stride=1, padding=0)
#         self.deconv_last = nn.ConvTranspose2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, stride=1, padding=0)
#         self.relu = nn.RReLU()

#     def forward(self, x):
#         residuals = []
#         layer = self.relu(self.conv_first(x))   #c1
#         residuals.append(layer.clone())
#         for _ in range(int(floor(self.depth-6)/2)):
#             for _ in range(2):
#                 layer = self.bn(layer)
#                 layer = self.relu(layer)
#                 layer = self.conv(layer)
#             residuals.append(layer.clone())
#         layer = self.relu(self.conv(layer))     #clast
#         layer = self.relu(self.deconv(layer))   #d1
#         layer = self.relu(layer+residuals.pop())
#         for _ in range(int(floor(self.depth-6)/2)):
#             for _ in range(2):
#                 layer = self.bn(layer)
#                 layer = self.relu(layer)
#                 layer = self.deconv(layer)
#             layer = self.relu(layer+residuals.pop())
#         layer = self.relu(self.deconv_last(layer))
#         return layer

# UNET from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
# sub-parts of the U-Net model


# class double_conv(nn.Module):
#     '''(conv => BN => ReLU) * 2'''
#     def __init__(self, in_ch, out_ch, relu):
#         super(double_conv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             relu(inplace=True),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             relu(inplace=True)
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         return x


# class inconv(nn.Module):
#     def __init__(self, in_ch, out_ch, relu):
#         super(inconv, self).__init__()
#         self.conv = double_conv(in_ch, out_ch, relu)

#     def forward(self, x):
#         x = self.conv(x)
#         return x


# class down(nn.Module):
#     def __init__(self, in_ch, out_ch, relu):
#         super(down, self).__init__()
#         self.mpconv = nn.Sequential(
#             nn.MaxPool2d(2),
#             double_conv(in_ch, out_ch, relu)
#         )

#     def forward(self, x):
#         x = self.mpconv(x)
#         return x


# class up(nn.Module):
#     def __init__(self, in_ch, out_ch, relu):
#         super(up, self).__init__()
#         self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

#         self.conv = double_conv(in_ch, out_ch, relu)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)

        # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
#                         diffY // 2, diffY - diffY//2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

#         x = torch.cat([x2, x1], dim=1)
#         x = self.conv(x)
#         return x


# class outconv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(outconv, self).__init__()
#         self.conv = nn.Conv2d(in_ch, out_ch, 1)

#     def forward(self, x):
#         x = self.conv(x)
#         return x

# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, relu='relu'):
#         super(UNet, self).__init__()
#         relu = nn.ReLU if relu == 'relu' else nn.RReLU
#         self.inc = inconv(n_channels, 64, relu)
#         self.down1 = down(64, 128, relu)
#         self.down2 = down(128, 256, relu)
#         self.down3 = down(256, 512, relu)
#         self.down4 = down(512, 512, relu)
#         self.up1 = up(1024, 256, relu)
#         self.up2 = up(512, 128, relu)
#         self.up3 = up(256, 64, relu)
#         self.up4 = up(128, 64, relu)
#         self.outc = outconv(64, n_classes)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         x = self.outc(x)
#         return F.sigmoid(x)

# UNET from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
# sub-parts of the U-Net model


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

############################################################################
## RReLU ugly copypaste

class rrdouble_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(rrdouble_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.RReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.RReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class rrinconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(rrinconv, self).__init__()
        self.conv = rrdouble_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class rrdown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(rrdown, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            rrdouble_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class rrup(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(rrup, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = rrdouble_conv(in_ch, out_ch)

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

class RRUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(RRUNet, self).__init__()
        self.inc = rrinconv(n_channels, 64)
        self.down1 = rrdown(64, 128)
        self.down2 = rrdown(128, 256)
        self.down3 = rrdown(256, 512)
        self.down4 = rrdown(512, 512)
        self.up1 = rrup(1024, 256)
        self.up2 = rrup(512, 128)
        self.up3 = rrup(256, 64)
        self.up4 = rrup(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
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
        return F.sigmoid(x)
