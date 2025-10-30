from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lmp
import lmp.nn as lnn
import lmp.nn.functional as LF
from lmp.util.typing import *

from .base import BaseClassifier


@lmp.register("classifier-linear")
class LinearClassifier(BaseClassifier):
    @dataclass
    class Config(BaseClassifier.Config):
        in_ch_0: int = 20
        in_S: int = 30
        in_dim: int = 2

        ch: int = 128

        out_dim: int = 10

        aux_ch_in: int = 0
        aux_ch: int = 64
        aux_dim: int = 2 # 2D or 3D

        flat_aux_feat_in: int = 0
        flat_aux_ch: int = 64

        dropout_in: float = 0.0
        dropout_feat: float = 0.0
        dropout_aux: float = 0.0
        dropout_flat_aux: float = 0.0

        input_act: bool = False # act on input x

    cfg: Config

    def configure(self) -> None:
        super().configure()

        in_ch = self.cfg.in_dim
        ch = self.cfg.ch
        out_dim = self.cfg.out_dim

        aux_ch_in = self.cfg.aux_ch_in
        self.has_aux = aux_ch_in >= 1
        if self.has_aux:
            aux_dim = self.cfg.aux_dim
            if aux_dim == 1:
                pool = nn.AdaptiveAvgPool1d
            elif aux_dim == 2:
                pool = nn.AdaptiveAvgPool2d
            elif aux_dim == 3:
                pool = nn.AdaptiveAvgPool3d
            else:
                pool = nn.Identity

            self.aux_net = nn.Sequential(
                nn.ReLU(),
                pool(1),
                nn.Flatten(),
                nn.Linear(aux_ch_in, self.cfg.aux_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(self.cfg.dropout_aux)
            )
        flat_aux_ch_in = self.cfg.flat_aux_feat_in
        self.has_flat_aux = flat_aux_ch_in >= 1
        if self.has_flat_aux:
            self.flat_aux_net = nn.Sequential(
               nn.ReLU(),
               nn.Linear(flat_aux_ch_in, self.cfg.flat_aux_ch),
               nn.ReLU(inplace=True),
               nn.Dropout(self.cfg.dropout_flat_aux)
            )
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * self.cfg.in_ch_0 * self.cfg.in_S, ch),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.dropout_in)
        )
        aux_ch = self.cfg.aux_ch if self.has_aux else 0
        flat_aux_ch = self.cfg.flat_aux_ch if self.has_flat_aux else 0
        self.head = nn.Sequential(
            nn.Linear(ch + aux_ch + flat_aux_ch, ch),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.dropout_feat),
            nn.Linear(ch, out_dim)
        )

        if self.cfg.input_act:
            self.input_act = nn.ReLU(inplace=True)

    def forward(self, G_p: Float[Tensor, "B C0 S dim"], aux_img: Optional[Float[Tensor, "B C *D"]], aux_flat: Optional[Float[Tensor, "B L"]]) -> Float[Tensor, "B Dout"]:
        if self.cfg.input_act:
            G_p = self.input_act(G_p)
        x = self.net(G_p)
        if self.has_aux:
            assert aux_img is not None
            aux_feat = self.aux_net(aux_img)
            x = torch.cat((x, aux_feat), dim=1)
        if self.has_flat_aux:
            assert aux_flat is not None
            aux_feat = self.flat_aux_net(aux_flat)
            x = torch.cat((x, aux_feat), dim=1)
        out = self.head(x)
        return out