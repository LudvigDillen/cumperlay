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


@lmp.register("persistence-image-representation")
class PersistenceImageRepresentation(BaseRepresentation):
    @dataclass
    class Config(BaseRepresentation.Config):
        weight_power: float = 2.0

        image_size: int = 32
        image_bnd_min: float = -0.5
        image_bnd_max: float = 1.5
        variance: float = .1

        out_dim: int = 128

    cfg: Config

    def configure(self) -> None:
        super().configure()

        image_size = self.cfg.image_size
        image_bnd_min = self.cfg.image_bnd_min
        image_bnd_max = self.cfg.image_bnd_max
        image_bounds = torch.tensor([[image_bnd_min, image_bnd_max], [image_bnd_min, image_bnd_max]], device=self.device)

        weight_fn = lnn.perslay.PowerPerslayWeight(constant=1.0, power=self.cfg.weight_power, learnable_constant=False, normalize=True)
        phi_fn = lnn.perslay.GaussianPerslayPhi([image_size, image_size], image_bounds, self.cfg.variance)
        reduce_op = torch.sum
        
        self.pers_image = lnn.perslay.Perslay(weight_fn, phi_fn, reduce_op)

        out_dim  = self.cfg.out_dim
        self.out_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, pers_diags: Float[Tensor, "B F 2"], diags_mask: Float[Tensor, "B F"]) -> Float[Tensor, "B S"]:
        B = pers_diags.shape[0]
        pers_image = self.pers_image(pers_diags, diags_mask)
        out_f = self.out_net(pers_image.unsqueeze(1)).view(B, -1)
        return out_f

    @property
    def tensor_type(self) -> str:
        return "flat"