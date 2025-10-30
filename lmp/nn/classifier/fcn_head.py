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


@lmp.register("classifier-fcn-head")
class FCNHead(BaseClassifier):
    @dataclass
    class Config(BaseClassifier.Config):
        in_channels: int = 0
        channels: int = 0

    cfg: Config

    def configure(self) -> None:
        super().configure()

        in_channels = self.cfg.in_channels
        channels = self.cfg.channels

        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
