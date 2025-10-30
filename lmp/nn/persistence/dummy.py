from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lmp
import lmp.nn as lnn
import lmp.nn.functional as LF
from lmp.util.typing import *

from .base import BasePersistence


@lmp.register("persistence-dummy-predictor")
class DummyPersistencePredictor(BasePersistence):
    @dataclass
    class Config(BasePersistence.Config):
        ch: int = 16
        out_s: int = 10
        out_l: int = 3
        dim: int = 2 # 2D or 3D

    cfg: Config

    def configure(self) -> None:
        super().configure()

        dim = self.cfg.dim
        if dim == 2:
            conv = nn.Conv2d
            norm = nn.BatchNorm2d
            pool = nn.AdaptiveMaxPool2d
        elif dim == 3:
            conv = nn.Conv3d
            norm = nn.BatchNorm3d
            pool = nn.AdaptiveAvgPool3d
        else:
            raise ValueError(f"DummyPersistancePredictor should be 2D or 3D, found dim: {dim}")

        self.out_s = self.cfg.out_s
        self.out_l = self.cfg.out_l

        self.net = nn.Sequential(
            conv(1, self.cfg.ch, kernel_size=3, padding=1),
            norm(self.cfg.ch),
            nn.ReLU(),
            pool(1),
            nn.Flatten(),
            nn.Linear(self.cfg.ch, self.out_s * self.out_l)
        )

    def forward(self, G_filt: Float[Tensor, "B C0 *D"], G_filt_unnorm: Float[Tensor, "B C0 *D"], G_ccl: Optional[Float[Tensor, "B C0 C1 *D"]]) -> Float[Tensor, "B C0 S L"]:
        B, C0, *D = G_filt.shape
        G_filt = G_filt.view(B*C0, 1, *D)
        out: Float[Tensor, "B*C0 S*L"] = self.net(G_filt)
        return out.view(B, C0, self.out_s, self.out_l), None
