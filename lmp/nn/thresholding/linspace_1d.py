from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lmp
import lmp.nn as lnn
import lmp.nn.functional as LF
from lmp.util.typing import *

from .base import BaseThresholding

@lmp.register("linspace-1d-thresholding")
class Linspace1DThresholding(BaseThresholding):
    @dataclass
    class Config(BaseThresholding.Config):
        threshold_grad_fn: str = "atan"

        n_thresholds: int = 20

        thres_max: float = 0.5
        thres_min: float = -0.5

        dim: int = 2 # 1 or 2

    cfg: Config

    thresholds: Float[Tensor, "F"]

    def configure(self) -> None:
        super().configure()

        self.threshold_fn = lnn.get_surrogate_grad_threshold_fn(self.cfg.threshold_grad_fn)

        self.F = self.cfg.n_thresholds
        self.register_buffer("thresholds", torch.linspace(self.cfg.thres_max, self.cfg.thres_min, self.F, device=self.device))

        dim = self.cfg.dim
        assert dim in [1, 2]
        if dim == 1:
            self.shape = [1, self.F, 1]
        elif dim == 2:
            self.shape = [1, 1, self.F]
    
    @property
    def ordered_thresholds(self):
        return self.thresholds

    def forward(self, G: Float[Tensor, "B C0 C1 *D"]) -> Float[Tensor, "B C0 C1 *D"]:
        B, C0, C1, *D = G.shape

        thresholds = self.ordered_thresholds.view(*self.shape, *[1 for _ in D])
        G_bin: Float[Tensor, "B C0 C1 *D"] = self.threshold_fn(G - thresholds)

        return G_bin
