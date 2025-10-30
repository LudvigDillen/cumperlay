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

@lmp.register("bifiltration-multith-erosion")
class MultiThBiFiltrationErosion(BaseFiltration):
    @dataclass
    class Config(BaseFiltration.Config):
        thresholding_type: str = ""
        thresholding: dict = field(default_factory=dict)

        bifilt_option: Optional[str] = "ic"

        n_channels_0: int = 20
        n_channels_1: int = 30
        erosion_method: str = "all_same" # all_same | row_same | column_same | unique
        erosion_softmax: bool = True

    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.thresholding = lmp.find(self.cfg.thresholding_type)(self.cfg.thresholding)

        n_channels_0 = self.cfg.n_channels_0
        n_channels_1 = self.cfg.n_channels_1
        method = self.cfg.erosion_method
        assert method in ["all_same", "row_same", "column_same", "unique"]
        soft_max=self.cfg.erosion_softmax

        if method == "all_same":
            self.erosion = lnn.Erosion2d(in_channels=1, out_channels=1, kernel_size=5, soft_max=soft_max)
        elif method == "row_same":
            self.erosion = lnn.Erosion2d(
                in_channels=n_channels_1, out_channels=n_channels_1, kernel_size=5, soft_max=soft_max
            )
        else:
            erosions = []
            for i in range(n_channels_0):     
                if method == "column_same":
                    erosion = lnn.Erosion2d(
                        in_channels=1, out_channels=1, kernel_size = 3 + 2 * (n_channels_0 - i - 1), soft_max=soft_max
                    )
                else: # if method == "unique":
                    erosion = lnn.Erosion2d(
                        in_channels=n_channels_1, out_channels=n_channels_1, kernel_size = 3 + 2 * (n_channels_0 - i - 1), soft_max=soft_max
                    )
                erosions.append(
                    erosion
                )
            self.erosions = nn.ModuleList(erosions)
        self.method = method

        self.has_ccl = False

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
        
        method = self.method
        G_er = []
        if method == "all_same":
            # -> B, C1, C0, *D ->
            G_curr = G.transpose(1, 2).contiguous().view(B * C1, C0, *D)
            for i in range(C0):
                BC_, C0_, *D_ = G_curr.shape
                # -> B*C1*(C0^), 1, *D -> B*C1, C0^, 1, *D
                G_curr = self.erosion(G_curr.view(BC_*C0_, 1, *D_)).view(BC_, C0_, *D_)
                # B*C1 1 *D -> B C1 *D
                G_er.insert(0, G_curr[:, -1].view(B, C1, *D))
                # B*C1 C0^ *D -> B*C1 (C0^-1) *D
                G_curr = G_curr[:, :-1]
        elif method == "row_same":
            G_curr = G.view(B * C0, C1, *D)
            for i in range(C0):
                G_curr = self.erosion(G_curr)
                G_curr = G_curr.view(B, -1, C1, *D)
                # B C0^ C1 *D -> B C1 *D
                G_er.insert(0, G_curr[:, -1])
                # B C0^ C1 *D -> B*(C0^-1) C1 *D
                G_curr = G_curr[:, :-1].view(-1, C1, *D)
        elif method == "column_same":
            # -> B, C1, C0, *D ->
            G = G.transpose(1, 2).contiguous().view(B * C1, C0, *D)
            for i, er in enumerate(self.erosions):
                # B*C1 1 *D -> B C1 *D
                G_er.append(er(G[:, i:i+1]).view(B, C1, *D))
        else: # if method == "unique":
            for i, er in enumerate(self.erosions):
                # B C1 *D -> B C1 *D
                G_er.append(er(G[:, i]))
        # C0 (B C1 *D) -> B C0 C1 *D
        G_er = torch.stack(G_er, dim=1)

        G_bin: Float[Tensor, "B C0 C1 *D"] = self.thresholding(G_er)

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

