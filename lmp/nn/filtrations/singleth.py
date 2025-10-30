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
from .functional import create_compact_single_filtration

@lmp.register("filtration-single-th")
class SingleThFiltration(BaseFiltration):
    @dataclass
    class Config(BaseFiltration.Config):
        thresholding_type: str = ""
        thresholding: dict = field(default_factory=dict)

    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.thresholding = lmp.find(self.cfg.thresholding_type)(self.cfg.thresholding)

    @property
    def ordered_thresholds(self):
        return self.thresholding.ordered_thresholds

    def forward(self, G: Float[Tensor, "B 1 C1 *D"]) -> tuple[Float[Tensor, "B 1 *D"], Float[Tensor, "B 1 *D"], list[list[Float[Tensor, "B 1 *D"]]]]:
        B, _, C1, *D = G.shape
       
        G_bin: Float[Tensor, "B 1 C1 *D"] = self.thresholding(G)

        G_bin_f: Float[Tensor, "B C1 *D"]  = G_bin.squeeze(1)

        # Create the single filtration as single grayscale output (with values in {1., C1.+1})
        G_out: Float[Tensor, "B 1 *D"]
        G_filt: list[Float[Tensor, "B 1 *D"]]
        G_out, G_filt = create_compact_single_filtration(G_bin_f)
        G_out_norm = G_out / C1 # [0, C1] -> [0, 1]

        G_filt = [G_filt]
        
        return G_out_norm, G_out, G_filt, None
