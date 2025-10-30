from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lmp
import lmp.nn as lnn
import lmp.nn.functional as LF
from lmp.util.typing import *

from .base import BaseNetwork


@lmp.register("network-unet-segmentation")
class UnetSegmentation(BaseNetwork):
    @dataclass
    class Config(BaseNetwork.Config):
        in_ch: int = 3
        out_ch: int = 64

        encoder_ch: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
        decoder_ch: list[int] = field(default_factory=lambda: [1024, 512, 256, 128])

    cfg: Config

    def configure(self) -> None:
        super().configure()

        ch = self.cfg.encoder_ch
        tr = self.cfg.decoder_ch

        self.in_ch = self.cfg.in_ch
        self.out_ch = self.cfg.out_ch

        # Input Convolution
        self.in_conv = DoubleConv(self.in_ch, ch[0])

        # Encoder path
        self.down1 = Down(ch[0], ch[1])
        self.down2 = Down(ch[1], ch[2])
        self.down3 = Down(ch[2], ch[3])
        self.down4 = Down(ch[3], tr[0])

        # Decoder path
        self.up4 = Up(tr[0], tr[1])
        self.up3 = Up(tr[1], tr[2])
        self.up2 = Up(tr[2], tr[3])
        self.up1 = Up(tr[3], self.out_ch)

    def forward(self, x: Float[Tensor, "B Cin H W"]) -> tuple[Float[Tensor, "B Cout0 Cout1 H W"], Float[Tensor, "B Cmid H W"]]:
        skip1 = self.in_conv(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        skip4 = self.down3(skip3)
        out = self.down4(skip4)

        out = self.up4(out, skip4) 
        out = self.up3(out, skip3) 
        out = self.up2(out, skip2) 
        out = self.up1(out, skip1) 

        return {
            "out": out
        }


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: Tensor, x2: Tensor):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
