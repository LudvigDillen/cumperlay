from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lmp
import lmp.nn as lnn
import lmp.nn.functional as LF
from lmp.util.typing import *

from torchvision.models import resnet

from .base import BaseNetwork
from .swin_v2 import SwinV2

@lmp.register("network-topo-swin-v2")
class TopoSwinV2(SwinV2):
    @dataclass
    class Config(SwinV2.Config):
        shared_topo: bool = True
        has_topo_layer: list[bool] = field(default_factory=lambda: [True, True, True, True])

        topo_ch: int = 2048

        gate_bias: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.topo_ch = self.cfg.topo_ch
        self.topo_linear = nn.Linear(self.topo_ch, self.num_classes)

        self.shared_topo = self.cfg.shared_topo
        self.has_topo_layer = self.cfg.has_topo_layer

    def update(self):
        gate_kwargs = {
            "bias": self.cfg.gate_bias
        }
        if self.has_topo_layer[0]:
            self.linear_1 = GatedTopoLinear(self.topo_ch, 128, **gate_kwargs)
        if self.has_topo_layer[1]:
            self.linear_2 = GatedTopoLinear(self.topo_ch, 256, **gate_kwargs)
        if self.has_topo_layer[2]:
            self.linear_3 = GatedTopoLinear(self.topo_ch, 512, **gate_kwargs)
        if self.has_topo_layer[3]:
            self.linear_4 = GatedTopoLinear(self.topo_ch, 1024, **gate_kwargs)

        self.patch_embedding = self.features[0]
        # print(self.patch_embedding[0].weight)
        self.stage_1 = self.features[1]
        self.stage_2 = nn.Sequential(*self.features[2:4])
        self.stage_3 = nn.Sequential(*self.features[4:6])
        self.stage_4 = nn.Sequential(*self.features[6:8])

        print(f"delete features =================================")
        del self.features

    def forward(self, x: Float[Tensor, "B Cin H W"], pd: Union[Float[Tensor, "B F"], Float[Tensor, "B 4 F"]]) -> Float[Tensor, "B Cout"]:
        x = self.patch_embedding(x)  # (3, 224, 224) -> (56, 56, 128)
        x = self.stage_1(x)  # (56, 56, 128) -> (56, 56, 128)

        if self.shared_topo:
            o = pd
        else:
            o = pd[:, 0]

        out_topo = o
        if self.has_topo_layer[0]:
            x, out1 = self.linear_1(x, o)

        # second topology
        x = self.stage_2(x)  # (56, 56, 128) -> (28, 28, 256)
        if not self.shared_topo:
            o = pd[:, 1]
        if self.has_topo_layer[1]:
            x, out2 = self.linear_2(x, o)

        # third topology
        x = self.stage_3(x)  # (28, 28, 256) -> (14, 14, 512)
        if not self.shared_topo:
            o = pd[:, 2]
        if self.has_topo_layer[2]:
            x, out3 = self.linear_3(x, o)

        # fourth topology
        x = self.stage_4(x)  # (14, 14, 512) -> (7, 7, 1024)
        if not self.shared_topo:
            o = pd[:, 3]
        if self.has_topo_layer[3]:
            x, out4 = self.linear_4(x, o)

        # x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)  # (7, 7, 1024) -> (1024, 7, 7)
        x = self.avgpool(x)  # (1024, 7, 7) -> (1024, 1, 1)
        x = self.flatten(x)  # (1024, 1, 1) -> (1024)
        x = self.head(x)  # (1024) -> (2)

        return x, self.topo_linear(out_topo)


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
