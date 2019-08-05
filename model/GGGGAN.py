import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from .UNET import *

class Generator(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(Generator, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1_1 = up(1024, 256)
        self.up1_2 = up(512, 128)
        self.up1_3 = up(256, 64)
        self.up1_4 = up(128, 64)
        self.out1_c = outconv(64, n_classes)
        self.up2_1 = up(1024, 256)
        self.up2_2 = up(512, 128)
        self.up2_3 = up(256, 64)
        self.up2_4 = up(128, 64)
        self.out2_c = outconv(64, n_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            #elif isinstance(m, nn.utils.spectral_norm):
            #    m.weight.data.normal_(1.0, 0.02)
            #    if m.bias is not None:
            #        m.bias.data.zero_()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1_1(x5, x4)
        x = self.up1_2(x, x3)
        x = self.up1_3(x, x2)
        x = self.up1_4(x, x1)
        x = self.out1_c(x)
        x = torch.tanh(x)
        y = self.up1_1(x5, x4)
        y = self.up1_2(y, x3)
        y = self.up1_3(y, x2)
        y = self.up1_4(y, x1)
        y = self.out1_c(y)
        y = torch.tanh(y)
        return x, y

class Discriminator(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(Discriminator, self).__init__()

        use_ins_norm = True

        self.model = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_ins_norm)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_ins_norm)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=use_ins_norm)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, n_classes, kernel_size=3, stride=1, padding=1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            #elif isinstance(m, nn.utils.spectral_norm):
            #    m.weight.data.normal_(1.0, 0.02)
            #    if m.bias is not None:
            #        m.bias.data.zero_()

    def forward(self, img):
        validity = self.model(img)

        return validity
