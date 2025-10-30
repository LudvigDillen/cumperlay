from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lmp
import lmp.nn as lnn
import lmp.nn.functional as LF
from lmp.util.typing import *

from .base import BaseRepresentation


@lmp.register("attention-representation")
class AttentionRepresentation(BaseRepresentation):
    @dataclass
    class Config(BaseRepresentation.Config):
        pd_dim: int = 2
        embd_dim: int = 16

        ch: int = 64
        out_ch: int = 512

    cfg: Config

    embd_t: Float[Tensor, "1 1 dim 1 E"]

    def configure(self) -> None:
        super().configure()

        self.mapping_fn = pers_tent_fn

        pd_dim = self.cfg.pd_dim
        embd_dim = self.cfg.embd_dim

        embd_t = torch.randn((1, 1, pd_dim, 1, embd_dim), device=self.device) * 0.02
        self.register_parameter("embd_t", nn.Parameter(embd_t)) # type: ignore

        self.embd_mlp = nn.Sequential(nn.Linear(2, embd_dim))

        ch = self.cfg.ch
        self.network = nn.Sequential(
            nn.Linear(embd_dim, ch),
            nn.ReLU(inplace=True),

            nn.Linear(ch, 2*ch),
            nn.ReLU(inplace=True),

            nn.Linear(2*ch, 4*ch),
            nn.ReLU(inplace=True),

            nn.Linear(4*ch, self.cfg.out_ch)
        )

        self.weight_network = nn.Sequential()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.embd_t, std=.02)

    def forward(self, pers_diags: Float[Tensor, "B C dim F 2"], diags_mask: Float[Tensor, "B C dim F"]) -> Float[Tensor, "B C*dim S"]:
        B, C0, dim, F0, _ = pers_diags.shape

        bd_t: Float[Tensor, "B C dim F 2"] = self.mapping_fn(pers_diags, diags_mask)
        pers_embed: Float[Tensor, "B C dim F E"]  = self.embd_mlp(bd_t) + self.embd_t

        return pers_embed

    @property
    def tensor_type(self) -> str:
        return "full"

def pers_tent_fn(pers_diags: Tensor, diags_mask: Tensor) -> Tensor:
    birth, death = pers_diags[..., 0], pers_diags[..., 1]
    return torch.stack((0.5 * (death - birth), 0.5 * (death + birth)), dim=-1)
