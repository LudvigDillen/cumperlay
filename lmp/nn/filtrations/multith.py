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
from .functional import create_compact_bifiltration, create_compact_bifiltration_ic, create_compact_bifiltration_ic_cat

@lmp.register("bifiltration-multi-th")
class MultiThBiFiltration(BaseFiltration):
    @dataclass
    class Config(BaseFiltration.Config):
        thresholding_type: str = ""
        thresholding: dict = field(default_factory=dict)

        bifilt_option: Optional[str] = "ic"

    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.thresholding = lmp.find(self.cfg.thresholding_type)(self.cfg.thresholding)

        self.has_ccl = False

        assert self.cfg.bifilt_option is None or self.cfg.bifilt_option in ["ic", "ic_cat", "ic_ccl"]
        self.compact_bifilt_fn = create_compact_bifiltration
        if self.cfg.bifilt_option == "ic":
            self.compact_bifilt_fn = create_compact_bifiltration_ic
        elif self.cfg.bifilt_option == "ic_cat":
            self.compact_bifilt_fn = create_compact_bifiltration_ic_cat
        elif self.cfg.bifilt_option == "ic_ccl":
            from .functional_ccl import create_compact_bifiltration_ic_ccl
            self.compact_bifilt_fn = create_compact_bifiltration_ic_ccl
            self.has_ccl = True

    @property
    def ordered_thresholds(self):
        return self.thresholding.ordered_thresholds

    def forward(self, G: Float[Tensor, "B C0 C1 *D"]) -> tuple[Float[Tensor, "B C0 *D"], Float[Tensor, "B C0 *D"], list[list[Float[Tensor, "B 1 *D"]]]]:
        B, C0, C1, *D = G.shape

        G_bin: Float[Tensor, "B C0 C1 *D"] = self.thresholding(G)

        # Create the filtrations as grayscale output (with values in {0., 1., C1.})
        G_out: Float[Tensor, "B C0 *D"] 
        G_filt: list[list[Float[Tensor, "B 1 *D"]]]
        out = self.compact_bifilt_fn(G_bin)
        if self.has_ccl:
            G_out, G_filt, G_ccl = out
        else:
            G_out, G_filt = out
            G_ccl = None
        G_out_norm = G_out / C1 # [0, C1] -> [0, 1]

        # print(G.shape, G_bin.shape, G_out.shape, G_out_norm.shape, (len(G_filt), len(G_filt[0]), G_filt[0][0].shape))
        
        return G_out_norm, G_out, G_filt, G_ccl

