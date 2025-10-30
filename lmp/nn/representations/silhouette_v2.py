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


@lmp.register("silhouette-v2-representation")
class SilhouetteV2Representation(BaseRepresentation):
    @dataclass
    class Config(BaseRepresentation.Config):
        weight_power: float = 1.0
        weight_learnable: bool = True

        n_samples: int = 128

        n_rows: int = 8
        n_dim: int = 2

    cfg: Config

    def configure(self) -> None:
        super().configure()

        n_rows = self.cfg.n_rows
        n_dim = self.cfg.n_dim
        n_S = self.cfg.n_samples

        self.weight_fn = MultidimPowerPerslayWeight(n_rows, n_dim, constant=1.0, power=self.cfg.weight_power, learnable_constant=True, learnable_power=True, normalize=True)
        self.phi_fn = MultidimTentPerslayPhi(n_rows, n_dim, n_S)
        self.reduce_op = torch.sum

    def forward(self, pers_diags: Float[Tensor, "B C dim F 2"], diags_mask: Float[Tensor, "B C dim F"]) -> Float[Tensor, "B C S dim"]:
        B, C0, dim, F0, _ = pers_diags.shape

        vector = self.phi_fn(pers_diags, diags_mask) # B C dim F S
        weight = self.weight_fn(pers_diags, diags_mask).unsqueeze(-1) # B C dim F 1
        vector = vector * weight # B C dim F S

        vector = self.reduce_op(vector, dim=3) # B C dim S

        return vector.permute(0, 1, 3, 2)

    @property
    def tensor_type(self) -> str:
        return "full"
    
class MultidimPowerPerslayWeight(nn.Module):
    constant: Union[Float[Tensor, "1 C dim 1"], float]
    power: Union[Float[Tensor, "1 C dim 1"], float]

    def __init__(self, n_ch: int, n_dim: int, constant: float, power: float, learnable_constant: bool = True, learnable_power: bool = True, normalize: bool = False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        shape = (1, n_ch, n_dim, 1)
        if learnable_constant:
            self.register_parameter("constant", nn.Parameter(
                torch.full(shape, constant, **factory_kwargs))) # type: ignore
        else:
            self.constant = constant
        if learnable_power:
            self.register_parameter("power", nn.Parameter(
                torch.full(shape, power, **factory_kwargs))) # type: ignore
        else:
            self.power = power
        
        self.normalize = normalize
        self.eps = 1e-6

    def forward(self, diagrams: Float[Tensor, "B C dim F 2"], masks: Float[Tensor, "B C dim F"]) -> Float[Tensor, "B C dim F"]:
        xs, ys = diagrams[..., 0], diagrams[..., 1] # B C dim F
        weight = masks * self.constant * torch.pow(torch.abs(ys - xs) + self.eps, self.power)
        
        if self.normalize:
            weight = weight / (torch.sum(weight, dim=1, keepdim=True) + self.eps) # Normalize over F dimension
        return weight
    
class MultidimTentPerslayPhi(nn.Module):
    samples: Float[Tensor, "1 C dim 1 S"]

    def __init__(self, n_ch: int, n_dim: int, n_samples: int):
        super().__init__()

        self.C = n_ch
        self.dim = n_dim
        self.S = n_samples

        samples = torch.linspace(0., 1.0, n_samples).view(1, 1, n_samples).repeat(n_ch, n_dim, 1).view(1, n_ch, n_dim, 1, n_samples)
        self.register_parameter("samples", nn.Parameter(
            samples, requires_grad=True)) # type: ignore

    def forward(self, diagrams: Float[Tensor, "B C dim F 2"], masks: Float[Tensor, "B C dim F"]) -> Float[Tensor, "B C dim F S"]:
        xs, ys = diagrams[..., 0:1], diagrams[..., 1:2] # B C dim F 1
        mask = masks.unsqueeze(-1)
        output = mask * F.relu(.5 * (ys - xs) - torch.abs(self.samples - .5 * (ys + xs)))
        return output