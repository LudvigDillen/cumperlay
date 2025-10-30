from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lmp
import lmp.nn as lnn
import lmp.nn.functional as LF
from lmp.util.typing import *

from .base import BaseClassifier


@lmp.register("classifier-convolutional")
class ConvolutionalClassifier(BaseClassifier):
    @dataclass
    class Config(BaseClassifier.Config):
        in_ch_0: int = 20
        in_S: int = 30
        in_dim: int = 2

        ch: int = 64

        out_dim: int = 10

    cfg: Config

    def configure(self) -> None:
        super().configure()

        in_ch = self.cfg.in_dim
        ch = self.cfg.ch
        out_dim = self.cfg.out_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            
            nn.Conv2d(ch, 2*ch, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(2*ch),
            nn.ReLU(),

            nn.Conv2d(2*ch, 2*ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*ch),
            nn.ReLU(),

            nn.Flatten(),
        )

        dim_feat = 2 * ch * (self.cfg.in_ch_0 // 2) * (self.cfg.in_S // 2)
        self.head = nn.Sequential(
            nn.Linear(dim_feat, ch),
            nn.ReLU(),
            nn.Linear(ch, out_dim)
        )


    def forward(self, G_p: Float[Tensor, "B C0 S dim"]) -> Float[Tensor, "B Dout"]:
        G_im: Float[Tensor, "B dim C0 S"] = G_p.permute(0, 3, 1, 2)
        x = self.cnn(G_im)
        x = self.head(x)
        return x