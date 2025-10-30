from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import FunctionCtx

import lmp
import lmp.nn as lnn
import lmp.nn.functional as LF
from lmp.util.typing import *

from .base import BaseThresholding

@lmp.register("stair-combined-thresholding")
class StairCombinedThresholding(BaseThresholding):
    @dataclass
    class Config(BaseThresholding.Config):
        n_thresholds: int = 20

        sigmoid: bool = True
        min_max_normalize: bool = False

    cfg: Config


    def configure(self) -> None:
        super().configure()

        self.F = self.cfg.n_thresholds

        self.thresh_fn = stair_act_identity_grad(self.F+1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass

    @property
    def ordered_thresholds(self):
        return []

    def forward(self, G: Float[Tensor, "B C0 *D"]) -> Float[Tensor, "B C0 *D"]:
        # Sigmoid -> 0-1
        if self.cfg.sigmoid:
            G = torch.sigmoid(G)
        if self.cfg.min_max_normalize:
            G_flat = G.flatten(start_dim=1)
            G_min, G_max = G_flat.min(dim=1, keepdim=True)[0], G_flat.max(dim=1, keepdim=True)[0]
            for i in range(G.dim()-2):
                G_min = G_min.unsqueeze(-1)
                G_max = G_max.unsqueeze(-1)
            G = (G - G_min) / (G_max - G_min + 1e-8)
            G = G.clamp(min=0)

        G_stair: Float[Tensor, "B C0 *D"] = self.thresh_fn(G)

        return G_stair


class StairActivationFunctionIdentityGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, input_: Tensor, N: int) -> Tensor:
        out = input_.mul(N).floor_().div_(N)
        return out

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> tuple[Tensor, None]:
        grad_input = grad_output.clone()
        return grad_input, None



def stair_act_identity_grad(N: int):
    N = N

    def inner(x):
        return StairActivationFunctionIdentityGrad.apply(x, N)

    return inner