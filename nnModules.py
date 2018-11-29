import torch.nn as nn
import torch.nn.init as init
from math import floor

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

class RedCNN(nn.Module):
    def __init__(self, n_channels=96, image_channels=3, depth=22):
        super(RedCNN, self).__init__()
        self.depth = depth
        self.conv_first = nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=5, stride=1, padding=0)
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=5, stride=1, padding=0)
        self.deconv = nn.ConvTranspose2d(n_channels, n_channels, kernel_size=5, stride=1, padding=0)
        self.deconv_last = nn.ConvTranspose2d(in_channels=n_channels, out_channels=image_channels, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        residuals = []
        residuals.append(x.clone())
        layer = self.relu(self.conv_first(x))
        for _ in range(floor((self.depth-4)/2)):
            layer = self.relu(self.conv(layer))
            layer = self.relu(self.conv(layer))
            residuals.append(layer.clone())
        layer = self.relu(self.conv(layer))
        layer = self.deconv(layer)
        layer += residuals.pop()
        for _ in range(floor((self.depth-4)/2)):
            layer = self.deconv(self.relu(layer))
            layer = self.deconv(self.relu(layer))
            layer += residuals.pop()
        
        layer = self.deconv_last(self.relu(layer))
        layer += residuals.pop()
        layer = self.relu(layer)
        return layer
        
        # Encoder
        residual1 = x.clone()
        layer = self.relu(self.conv_first(x))
        layer = self.relu(self.conv(layer))
        residual2 = layer.clone()
        layer = self.relu(self.conv(layer))
        layer = self.relu(self.conv(layer))
        residual3 = layer.clone()
        layer = self.relu(self.conv(layer))
        # decoder
        layer = self.deconv(layer)
        layer += residual3
        layer = self.deconv(self.relu(layer))
        layer = self.deconv(self.relu(layer))
        layer += residual2
        layer = self.deconv(self.relu(layer))
        layer = self.deconv_last(self.relu(layer))
        layer += residual1
        layer = self.relu(layer)
        return layer
