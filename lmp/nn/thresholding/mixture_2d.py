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

@lmp.register("mixture-2d-thresholding")
class Mixture2DThresholding(BaseThresholding):
    @dataclass
    class Config(BaseThresholding.Config):
        threshold_grad_fn: str = "atan"

        n_thresholds_1: int = 20
        n_thresholds_2: int = 30

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

        self.F1 = self.cfg.n_thresholds_1
        self.F2 = self.cfg.n_thresholds_2

        self.register_buffer("sample_pts_1", torch.linspace(1-0.01, 0.01, self.F1, device=self.device))
        self.register_buffer("sample_pts_2", torch.linspace(1-0.01, 0.01, self.F2, device=self.device))

        self.register_parameter("temp", nn.Parameter(torch.ones(2, device=self.device)))
        self.register_parameter("w1", nn.Parameter(torch.zeros((1, 4), device=self.device)))
        self.register_parameter("b1", nn.Parameter(torch.zeros(1, device=self.device)))
        self.register_parameter("w2", nn.Parameter(torch.zeros((1, 4), device=self.device)))
        self.register_parameter("b2", nn.Parameter(torch.zeros(1, device=self.device)))

        self.register_parameter("scale", nn.Parameter(torch.scalar_tensor(self.cfg.scale_init, device=self.device)))
        self.register_parameter("offset", nn.Parameter(torch.scalar_tensor(self.cfg.offset_init, device=self.device)))

        self.scale_max = self.cfg.scale_max
        self.scale_min = self.cfg.scale_min

        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 0.01

        nn.init.normal_(self.b1, std=std)
        nn.init.normal_(self.b2, std=std)

    @property
    def ordered_thresholds(self):
        T1 = get_thresholds(self.w1, self.b1, self.sample_pts_1, self.temp[0])
        T2 = get_thresholds(self.w2, self.b2, self.sample_pts_2, self.temp[1])
        return T1, T2
    
    def forward(self, G: Float[Tensor, "B C0 C1 *D"]) -> Float[Tensor, "B C0 C1 *D"]:
        B, C0, C1, *D = G.shape

        G = (self.scale.clamp(min=self.scale_min, max=self.scale_max)) * G + self.offset
        
        T1, T2 = self.ordered_thresholds
        T1 = T1.view(1, self.F1, 1, *[1 for _ in D])
        T2 = T2.view(1, 1, self.F2, *[1 for _ in D])
        T = torch.maximum(T1, T2)
        G_bin: Float[Tensor, "B C0 C1 *D"] = self.threshold_fn(G - T)

        return G_bin

def get_thresholds(w: Tensor, b: Tensor, sample_pts: Tensor, temp: Tensor) -> Tensor:
    x = torch.stack((
        sample_pts,
        torch.sigmoid(temp*(sample_pts - 0.5)),
        torch.square(sample_pts),
        torch.sqrt(sample_pts)
    ), dim=-1) - 0.5
    out = F.linear(x, w.exp(), b)
    return out.squeeze(-1)