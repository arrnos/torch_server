import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.utils import model_zoo
from torchvision import models

# 编码块
class UNetEnc(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, dilation=2),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers += [nn.Dropout(.5)]
        layers += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        return self.down(x)


# 解码块
class UNetDec(nn.Module):

    def __init__(self, in_channels, features, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.Conv2d(in_channels, features, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features, out_channels, 2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


# U-Net
class UNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.enc1 = UNetEnc(3, 64)
        self.enc2 = UNetEnc(64, 128)
        self.enc3 = UNetEnc(128, 256)
        self.enc4 = UNetEnc(256, 512, dropout=True)
        self.center = nn.Sequential(
            nn.Conv2d(512, 1024, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.ConvTranspose2d(1024, 512, 2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.dec4 = UNetDec(1024, 512, 256)
        self.dec3 = UNetDec(512, 256, 128)
        self.dec2 = UNetDec(256, 128, 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, 1)

    # 前向传播过程
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        # 包含了同层分辨率级联的解码块
        dec4 = self.dec4(torch.cat([
            center, F.upsample_bilinear(enc4, center.size()[2:])], 1))
        dec3 = self.dec3(torch.cat([
            dec4, F.upsample_bilinear(enc3, dec4.size()[2:])], 1))
        dec2 = self.dec2(torch.cat([
            dec3, F.upsample_bilinear(enc2, dec3.size()[2:])], 1))
        dec1 = self.dec1(torch.cat([
            dec2, F.upsample_bilinear(enc1, dec2.size()[2:])], 1))

        return F.upsample_bilinear(self.final(dec1), x.size()[2:])