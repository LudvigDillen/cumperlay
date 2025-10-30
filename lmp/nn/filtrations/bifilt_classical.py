from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale

import lmp
import lmp.nn as lnn
import lmp.nn.functional as LF
from lmp.util.typing import *

from kornia.morphology import erosion

from .base import BaseFiltration
from . import functional_combined as fc

@lmp.register("bifiltration-classical")
class BiFiltrationClassical(BaseFiltration):
    @dataclass
    class Config(BaseFiltration.Config):
        thresholding_type: str = ""
        thresholding: dict = field(default_factory=dict)

        n_erosions: int = 10
        n_thresholds: int = 20
        
        in_ch: int = 3

        grayscale: bool = True
        unnorm: bool = True
        unnorm_mean: list[int] = field(default_factory=lambda: [0.5, 0.5, 0.5])
        unnorm_std: list[int] = field(default_factory=lambda: [0.5, 0.5, 0.5])

        kernel_size: int = 3

    cfg: Config

    unnorm_mean: Tensor
    unnorm_std: Tensor

    kernel: Tensor

    def configure(self) -> None:
        super().configure()

        self.N = self.cfg.n_erosions
        self.F = self.cfg.n_thresholds
        self.in_ch = self.cfg.in_ch

        self.thresholding = lmp.find(self.cfg.thresholding_type)(self.cfg.thresholding)

        if self.cfg.unnorm:
            self.register_buffer("unnorm_mean", torch.tensor(self.cfg.unnorm_mean, device=self.device).view(1, self.in_ch, 1, 1))
            self.register_buffer("unnorm_std", torch.tensor(self.cfg.unnorm_std, device=self.device).view(1, self.in_ch, 1, 1))

        kernel_size = self.cfg.kernel_size
        self.register_buffer("kernel", torch.ones((kernel_size, kernel_size), device=self.device))

    @property
    def ordered_thresholds(self):
        return self.thresholding.ordered_thresholds

    def forward(self, G: Float[Tensor, "B C0 *D"]) -> tuple[Float[Tensor, "B C0 *D"], Float[Tensor, "B C0 *D"], list[list[Float[Tensor, "B 1 *D"]]]]:
        B, C0, *D = G.shape

        if self.cfg.unnorm:
            G = G * self.unnorm_std + self.unnorm_mean
        if self.cfg.grayscale:
            G = rgb_to_grayscale(G) # B 1 *D
        
        outputs = [G]
        curr = G
        for iteration in range(self.cfg.n_erosions-1):
            new = erosion(curr, self.kernel, engine="convolution")
            outputs.append(new)
            curr = new

        G_eroded = torch.cat(outputs, dim=1)
        G_out_norm = self.thresholding(G_eroded)
        
        G_out = G_out_norm * self.F # [0, 1] -> [0, C1]

        # print(G.shape, G_bin.shape, G_out.shape, G_out_norm.shape, (len(G_filt), len(G_filt[0]), G_filt[0][0].shape))
        
        return G_out_norm, G_out
