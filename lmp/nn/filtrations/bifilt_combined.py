from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lmp
import lmp.nn as lnn
import lmp.nn.functional as LF
from lmp.util.typing import *

from .base import BaseFiltration
from . import functional_combined as fc

@lmp.register("bifiltration-combined")
class BiFiltrationCombined(BaseFiltration):
    @dataclass
    class Config(BaseFiltration.Config):
        thresholding_type: str = ""
        thresholding: dict = field(default_factory=dict)

        combine_method: str = "regular" # regular | ste | bmedian | bgaussian | disable
        straight_through: bool = False # old
        disable_bifilt_mapping: bool = False # old

        n_thresholds: int = 20

        bfilter_kernel_size: int = 3
        bgaussian_sigma: float = 1.0

    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.F = self.cfg.n_thresholds

        self.thresholding = lmp.find(self.cfg.thresholding_type)(self.cfg.thresholding)

        method = self.cfg.combine_method
        assert method in ["regular", "ste", "bmedian", "bgaussian", "disable"]
        if method == "regular" and self.cfg.straight_through:
            method = "ste"
        if self.cfg.disable_bifilt_mapping:
            method = "disable"
        
        if method == "regular":
            self.compact_bifilt_fn = fc.combine_compact_filtration
        elif method == "ste":
            self.compact_bifilt_fn = fc.combine_compact_filtration_ste_fn()
        elif method == "bmedian":
            self.compact_bifilt_fn = fc.combine_compact_filtration_b_median_filter(self.cfg.bfilter_kernel_size)
        elif method == "bgaussian":
            self.compact_bifilt_fn = fc.combine_compact_filtration_b_gaussian_filter(self.cfg.bfilter_kernel_size, self.cfg.bgaussian_sigma)
        elif method == "disable":
            self.compact_bifilt_fn = lambda x: x
        else:
            raise ValueError(f"Unknown method {method}")

    @property
    def ordered_thresholds(self):
        return self.thresholding.ordered_thresholds

    def forward(self, G: Float[Tensor, "B C0 *D"]) -> tuple[Float[Tensor, "B C0 *D"], Float[Tensor, "B C0 *D"], list[list[Float[Tensor, "B 1 *D"]]]]:
        B, C0, *D = G.shape

        G_bin: Float[Tensor, "B C0 *D"] = self.thresholding(G)
        G_out_norm: Float[Tensor, "B C0 *D"] = self.compact_bifilt_fn(G_bin)
        
        G_out = G_out_norm * self.F # [0, 1] -> [0, C1]

        # print(G.shape, G_bin.shape, G_out.shape, G_out_norm.shape, (len(G_filt), len(G_filt[0]), G_filt[0][0].shape))
        
        return G_out_norm, G_out

