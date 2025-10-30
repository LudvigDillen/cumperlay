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

@lmp.register("softplus-1d-thresholding")
class Softplus1DThresholding(BaseThresholding):
    @dataclass
    class Config(BaseThresholding.Config):
        threshold_grad_fn: str = "atan"

        n_thresholds: int = 20
        threshold_offset_init: float = 2.0

        softplus_beta: float = 5.0
        softplus_offset: float = 0.0

        scale_init: float = 0.1
        offset_init: float = 0.0

        scale_max: float = 2.0
        scale_min: float = 0.01

    cfg: Config

    threshold_offset: Float[Tensor, ""]
    thresholds: Float[Tensor, "F"]

    scale: Float[Tensor, ""]
    offset: Float[Tensor, ""]

    def configure(self) -> None:
        super().configure()

        self.threshold_fn = lnn.get_surrogate_grad_threshold_fn(self.cfg.threshold_grad_fn)

        self.softplus_beta = self.cfg.softplus_beta
        self.softplus_offset = self.cfg.softplus_offset
        self.F = self.cfg.n_thresholds

        self.register_parameter("threshold_offset", nn.Parameter(torch.scalar_tensor(self.cfg.threshold_offset_init, device=self.device)))
        self.register_parameter("thresholds", nn.Parameter(torch.empty((self.F), device=self.device)))

        self.register_parameter("scale", nn.Parameter(torch.scalar_tensor(self.cfg.scale_init, device=self.device)))
        self.register_parameter("offset", nn.Parameter(torch.scalar_tensor(self.cfg.offset_init, device=self.device)))

        self.scale_max = self.cfg.scale_max
        self.scale_min = self.cfg.scale_min

        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 0.01

        nn.init.normal_(self.thresholds, std=std)

    @property
    def ordered_thresholds(self):
        return self.threshold_offset + (-F.softplus(self.thresholds + self.softplus_offset, beta=self.softplus_beta)).cumsum(dim=0)

    def forward(self, G: Float[Tensor, "B C0 C1 *D"]) -> Float[Tensor, "B C0 C1 *D"]:
        # # Sigmoid -> 0-1
        # G = torch.sigmoid(G)

        G = (self.scale.clamp(min=self.scale_min, max=self.scale_max)) * G + self.offset

        # Do thresholding on the second channel dimension, ignore the first
        B, _, C1, *D = G.shape
        thresholds = self.ordered_thresholds.view(1, 1, self.F, *[1 for _ in D])
        G_bin: Float[Tensor, "B C0 C1 *D"] = self.threshold_fn(G - thresholds)

        return G_bin
