import torch.nn as nn
import torch.nn.init as init

class DnCNN(nn.Module):
    def __init__(self, depth=22, n_channels=64, image_channels=3, use_bnorm=True, kernel_size=3, find_noise=True):
        super(DnCNN, self).__init__()
        padding = (kernel_size/2)-1
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
    def __init__(self):
        super(RedCNN, self).__init__()
        layers = []
        self.redcnn = nn.Sequential(*layers)
        self.__initialize_weights()
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
