from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lmp
import lmp.nn as lnn
import lmp.nn.functional as LF
from lmp.util.typing import *

from lmp.nn.modules.blocks2d import PreActResBlock2d, down_block2d, up_block2d, norm_nonl_conv2d

from .base import BaseNetwork


@lmp.register("network-gridunet2d")
class GridUnet2d(BaseNetwork):
    @dataclass
    class Config(BaseNetwork.Config):
        in_ch: int = 3
        out_ch_0: int = 20
        out_ch_1: int = 30

        encoder_ch: list[int] = field(default_factory=lambda: [64, 128, 256])
        decoder_ch: list[int] = field(default_factory=lambda: [256, 256, 128])
        ch_last: int = 256

        downscale_factor_log2: int = 0

        bias: bool = True

    cfg: Config

    def configure(self) -> None:
        super().configure()

        ch = self.cfg.encoder_ch
        tr = self.cfg.decoder_ch

        self.in_ch = self.cfg.in_ch
        self.out_ch_0 = self.cfg.out_ch_0
        self.out_ch_1 = self.cfg.out_ch_1
        ch_last = self.cfg.ch_last

        bias = self.cfg.bias
        # Input Convolution
        self.conv1 = nn.Conv2d(self.in_ch, ch[0], kernel_size=5,
                               padding=5 // 2, bias=bias)
        # Encoder path
        self.enc_block1 = PreActResBlock2d(ch[0], 3, 1, bias)
        self.down1 = down_block2d(ch[0], ch[1], 3, bias)

        self.enc_block2 = PreActResBlock2d(ch[1], 3, 2, bias)
        self.down2 = down_block2d(ch[1], ch[2], 3, bias)

        self.enc_block3 = PreActResBlock2d(ch[2], 3, 2, bias)
        self.down3 = down_block2d(ch[2], ch[3], 3, bias)

        # Middle convolution 256
        self.mid_block = PreActResBlock2d(ch[3], 3, 3, bias)

        # Decoder path
        self.up3 = up_block2d(tr[0], tr[1] - ch[2], 3, bias)
        self.dec_block3 = PreActResBlock2d(tr[1], 3, 2, bias)

        self.up2 = up_block2d(tr[1], tr[2] - ch[1], 3, bias)
        self.dec_block2 = PreActResBlock2d(tr[2], 3, 2, bias)

        self.up1 = up_block2d(tr[2], tr[3] - ch[0], 3, bias)
        self.dec_block1 = PreActResBlock2d(tr[3], 3, 1, bias)

        self.out_conv = nn.Sequential(
            norm_nonl_conv2d(tr[2], ch_last,
                             kernel_size=3, padding=1, bias=bias),
            norm_nonl_conv2d(ch_last, self.out_ch_0 *
                             self.out_ch_1, kernel_size=3, padding=1, bias=bias),
        )
        self.out_down = nn.Sequential(
            *(down_block2d(self.out_ch_0 * self.out_ch_1, self.out_ch_0 * self.out_ch_1, 3, bias=bias) for _ in range(self.cfg.downscale_factor_log2))
        )

    def forward(self, x: Float[Tensor, "B Cin H W"]) -> tuple[Float[Tensor, "B Cout0 Cout1 H W"], Float[Tensor, "B Cmid H W"]]:
        enc0 = self.conv1(x)

        skip1 = self.enc_block1(enc0)
        enc1 = self.down1(skip1)

        skip2 = self.enc_block2(enc1)
        enc2 = self.down2(skip2)

        skip3 = self.enc_block3(enc2)
        enc3 = self.down3(skip3)

        mid = self.mid_block(enc3)

        dec3_lower = self.up3(mid)
        dec3 = torch.cat((dec3_lower, skip3), dim=1)
        out3 = self.dec_block3(dec3)

        dec2_lower = self.up2(out3)
        dec2 = torch.cat((dec2_lower, skip2), dim=1)
        out2 = self.dec_block2(dec2)

        dec1_lower = self.up1(out2)
        dec1 = torch.cat((dec1_lower, skip1), dim=1)
        out1 = self.dec_block1(dec1)

        out: Tensor = self.out_conv(out1)
        out = self.out_down(out)
        S = out.shape
        return out.view(S[0], self.out_ch_0, self.out_ch_1, *S[2:]), mid
