from dataclasses import dataclass, field
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lmp
import lmp.nn as lnn
import lmp.nn.functional as LF
from lmp.util.typing import *

from torchvision.ops.misc import Permute

from .base import BaseNetwork
from .swin_v2 import SwinV2

@lmp.register("network-mpb-swin-v2")
class MPBSwinV2(SwinV2):
    @dataclass
    class Config(SwinV2.Config):
        topo_ch: int = 256

        gate_bias: bool = True

        betti_inp_ch: int = 1000
        betti_ch: int = 128

    cfg: Config

    def configure(self) -> None:
        self.topo_ch = self.cfg.topo_ch

        super().configure()

        self.topo_net = nn.Sequential(
            nn.Linear(self.cfg.betti_ch, self.topo_ch),
            nn.ReLU(inplace=True),
            nn.Linear(self.topo_ch, self.num_classes)
        )

    def update(self):
        self.betti_net = BettiNet(self.cfg.betti_inp_ch, self.cfg.betti_ch)
        
        gate_kwargs = {
            "bias": self.cfg.gate_bias
        }
        self.lin_inp_1 = nn.Sequential(nn.Linear(self.cfg.betti_ch, self.topo_ch), nn.ReLU(inplace=True))
        self.linear_1 = GatedTopoLinear(self.topo_ch, 128, **gate_kwargs)
        self.lin_inp_2 = nn.Sequential(nn.Linear(self.cfg.betti_ch, self.topo_ch), nn.ReLU(inplace=True))
        self.linear_2 = GatedTopoLinear(self.topo_ch, 256, **gate_kwargs)
        self.lin_inp_3 = nn.Sequential(nn.Linear(self.cfg.betti_ch, self.topo_ch), nn.ReLU(inplace=True))
        self.linear_3 = GatedTopoLinear(self.topo_ch, 512, **gate_kwargs)
        self.lin_inp_4 = nn.Sequential(nn.Linear(self.cfg.betti_ch, self.topo_ch), nn.ReLU(inplace=True))
        self.linear_4 = GatedTopoLinear(self.topo_ch, 1024, **gate_kwargs)

        self.patch_embedding = self.features[0]
        # print(self.patch_embedding[0].weight)
        self.stage_1 = self.features[1]
        self.stage_2 = nn.Sequential(*self.features[2:4])
        self.stage_3 = nn.Sequential(*self.features[4:6])
        self.stage_4 = nn.Sequential(*self.features[6:8])

        print(f"delete features =================================")
        del self.features

    def forward(self, x: Float[Tensor, "B Cin H W"], betti: Float[Tensor, "B Cd *F"]) -> Tuple[Float[Tensor, "B Cout"], Float[Tensor, "B Cout"], None]:
        inp = x
        mpb = self.betti_net(betti)

        x = self.patch_embedding(x)  # (3, 224, 224) -> (56, 56, 128)
        x = self.stage_1(x)  # (56, 56, 128) -> (56, 56, 128)
        
        x, _ = self.linear_1(x, self.lin_inp_1(mpb))
        
        # second topology
        x = self.stage_2(x)  # (56, 56, 128) -> (28, 28, 256)
        x, _ = self.linear_2(x, self.lin_inp_2(mpb))

        # third topology
        x = self.stage_3(x)  # (28, 28, 256) -> (14, 14, 512)
        x, _ = self.linear_3(x, self.lin_inp_3(mpb))

        # fourth topology
        x = self.stage_4(x)  # (14, 14, 512) -> (7, 7, 1024)
        x, _ = self.linear_4(x, self.lin_inp_4(mpb))

        # x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)  # (7, 7, 1024) -> (1024, 7, 7)
        x = self.avgpool(x)  # (1024, 7, 7) -> (1024, 1, 1)
        x = self.flatten(x)  # (1024, 1, 1) -> (1024)
        x = self.head(x)  # (1024) -> (2)

        topo_out = self.topo_net(mpb)

        return x, topo_out, None


class BettiNet(nn.Module):
    def __init__(self, inp_ch, ch):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(inp_ch, 2*ch),
            nn.ReLU(inplace=True),
            nn.Linear(2*ch, ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class GatedTopoLinear(nn.Module):
    def __init__(self, in_ch, out_ch, bias=False):
        super().__init__()

        self.bias = bias
        mult = 2 if self.bias else 1
        self.lin = nn.Linear(in_ch, mult * out_ch)

    def forward(self, x, topo):
        out = self.lin(topo)
        if self.bias:
            scale, bias = F.sigmoid(out).chunk(2, dim=-1)
            scale = F.sigmoid(scale).unsqueeze(1).unsqueeze(2).expand_as(x)
            bias = bias.unsqueeze(1).unsqueeze(2).expand_as(x)
            out = x * scale + bias            
        else:
            scale = F.sigmoid(out).unsqueeze(1).unsqueeze(2).expand_as(x)
            out = x * scale
        x = x + out
        return x, out
