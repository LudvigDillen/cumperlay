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


@lmp.register("silhouette-representation")
class SilhouetteRepresentation(BaseRepresentation):
    @dataclass
    class Config(BaseRepresentation.Config):
        weight_power: float = 1.0

        n_samples: int = 30

    cfg: Config

    def configure(self) -> None:
        super().configure()

        initial_samples = torch.linspace(0., 1.0, self.cfg.n_samples)

        weight_fn = lnn.perslay.PowerPerslayWeight(constant=1.0, power=self.cfg.weight_power, learnable_constant=False, normalize=True)
        phi_fn = lnn.perslay.TentPerslayPhi(initial_samples)
        reduce_op = torch.sum
        
        self.silhouette = lnn.perslay.Perslay(weight_fn, phi_fn, reduce_op)

    def forward(self, pers_diags: Float[Tensor, "B F 2"], diags_mask: Float[Tensor, "B F"]) -> Float[Tensor, "B S"]:
        return self.silhouette(pers_diags, diags_mask)

    @property
    def tensor_type(self) -> str:
        return "flat"