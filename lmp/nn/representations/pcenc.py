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


@lmp.register("pcenc-representation")
class PointCloudEncoderRepresentation(BaseRepresentation):
    @dataclass
    class Config(BaseRepresentation.Config):
        pd_dim: int = 2
        embd_dim: int = 16
        embd_ch: int = 1

        ch: int = 64
        out_ch: int = 2048

        mapping: str = "tent" # tent | identity
        weight: str = "power" # power | identity
        
        weight_power: float = 1.0
        weight_learnable_power: bool = True

    cfg: Config

    embd_t: Float[Tensor, "1 Ce dim 1 E"]

    def configure(self) -> None:
        super().configure()

        mapping = self.cfg.mapping
        if mapping == "tent":
            self.mapping_fn = pers_tent_fn
        elif mapping == "identity":
            self.mapping_fn = pers_identity_fn
        else:
            raise ValueError(f"Unknown mapping {mapping}")
        
        weight = self.cfg.weight
        if weight == "identity":
            self.weight_fn = weight_identity_fn
        elif weight == "power":
            self.weight_fn = WeightTentFn(constant=1.0, power=self.cfg.weight_power, learnable_constant=False, learnable_power=self.cfg.weight_learnable_power)
        else:
            raise ValueError(f"Unknown weighting {weight}")

        pd_dim = self.cfg.pd_dim
        embd_dim = self.cfg.embd_dim
        embd_ch = self.cfg.embd_ch

        embd_t = torch.randn((1, embd_ch, pd_dim, 1, embd_dim), device=self.device) * 0.02
        self.register_parameter("embd_t", nn.Parameter(embd_t)) # type: ignore

        self.embd_mlp = nn.Linear(2, embd_dim)

        self.network = PointNetEncoder(embd_dim, self.cfg.ch, self.cfg.out_ch)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.embd_t, std=.02)
    
    def forward(self, pers_diags: Float[Tensor, "B C dim F 2"], diags_mask: Float[Tensor, "B C dim F"]) -> Float[Tensor, "B S"]:
        B, C0, dim, F0, _ = pers_diags.shape

        bd_t: Float[Tensor, "B C dim F 2"] = self.mapping_fn(pers_diags, diags_mask)
        pers_embd: Float[Tensor, "B C dim F E"]  = self.embd_mlp(bd_t) + self.embd_t
        
        x: Float[Tensor, "B E N"] = pers_embd.view(B, C0*dim*F0, -1).permute(0, 2, 1)
        weights: Float[Tensor, "B 1 N"] = self.weight_fn(pers_diags, diags_mask).view(B, 1, C0*dim*F0)
        
        out = self.network(x, weights)
        return out

    @property
    def tensor_type(self) -> str:
        return "full"

class PointNetEncoder(nn.Module):
    def __init__(self, in_ch, ch = 64, out_ch = 2048, eps=1e-6):
        super().__init__()

        self.conv_net = nn.Sequential(
            nn.Conv1d(in_ch, ch, 1),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),

            nn.Conv1d(ch, 2*ch, 1),
            nn.BatchNorm1d(2*ch),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(2*ch, 4*ch, 1),
            nn.BatchNorm1d(4*ch),
            nn.ReLU(inplace=True),

            nn.Conv1d(4*ch, out_ch, 1),
            # nn.BatchNorm1d(out_ch),

            # nn.Conv1d(4*ch, 8*ch, 1),
            # nn.BatchNorm1d(8*ch),
            # nn.ReLU(inplace=True),

            # nn.Conv1d(8*ch, out_ch, 1),
            # nn.BatchNorm1d(out_ch),

            # nn.Conv1d(8*ch, 16*ch, 1),
            # nn.BatchNorm1d(16*ch),
            # nn.ReLU(inplace=True),

            # nn.Conv1d(16*ch, out_ch, 1),
            # nn.BatchNorm1d(out_ch),
        )

        self.out_ch = out_ch

        self.eps = eps

    def forward(self, x: Float[Tensor, "B F N"], weights: Float[Tensor, "B 1 N"]) -> Float[Tensor, "B S"]:
        B, F0, N = x.shape

        x = self.conv_net(x)

        x = (weights * x).sum(dim=2) / (weights.sum(dim=2) + self.eps)
        # x = torch.max(x, 2, keepdim=True)[0]

        x = x.view(B, self.out_ch)
        return x

def pers_identity_fn(pers_diags: Tensor, diags_mask: Tensor) -> Tensor:
    return pers_diags

def pers_tent_fn(pers_diags: Tensor, diags_mask: Tensor) -> Tensor:
    birth, death = pers_diags[..., 0], pers_diags[..., 1]
    return torch.stack((0.5 * (death - birth), 0.5 * (death + birth)), dim=-1)

def weight_identity_fn(pers_diags: Tensor, diags_mask: Tensor) -> Tensor:
    return diags_mask

class WeightTentFn(nn.Module):
    constant: Union[Float[Tensor, ""], float]
    power: Union[Float[Tensor, ""], float]

    def __init__(self, constant: float, power: float, learnable_constant: bool = True, learnable_power: bool = True, device=None, dtype=None):

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if learnable_constant:
            self.register_parameter("constant", nn.Parameter(
                torch.scalar_tensor(constant, **factory_kwargs))) # type: ignore
        else:
            self.constant = constant
        if learnable_power:
            self.register_parameter("power", nn.Parameter(
                torch.scalar_tensor(power, **factory_kwargs))) # type: ignore
        else:
            self.power = power
        
        self.eps = 1e-6

    def forward(self, pers_diags: Tensor, diags_mask: Tensor) -> Tensor:
        birth, death = pers_diags[..., 0], pers_diags[..., 1]
        weight = self.constant * \
            torch.pow(torch.abs(death - birth) + self.eps, self.power)
        weight = weight * diags_mask
        
        return weight

