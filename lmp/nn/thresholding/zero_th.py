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

@lmp.register("zero-thresholding")
class ZeroThresholding(BaseThresholding):
    @dataclass
    class Config(BaseThresholding.Config):
        threshold_grad_fn: str = "atan"

        n_thresholds: int = 20

    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.threshold_fn = lnn.get_surrogate_grad_threshold_fn(self.cfg.threshold_grad_fn)

        self.F = self.cfg.n_thresholds

    @property
    def ordered_thresholds(self):
        return 0.0

    def forward(self, G: Float[Tensor, "B C0 C1 *D"]) -> Float[Tensor, "B C0 C1 *D"]:
        return self.threshold_fn(G)
