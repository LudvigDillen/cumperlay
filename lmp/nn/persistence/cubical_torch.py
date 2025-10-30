from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lmp
import lmp.nn as lnn
import lmp.nn.functional as LF
from lmp.util.typing import *

from .base import BasePersistence
from ..representations.base import BaseRepresentation

import cmp.cubical as cubical

@lmp.register("persistence-cubical-torch")
class CubicalPersistence(BasePersistence):
    @dataclass
    class Config(BasePersistence.Config):
        dim: int = 2 # 2 or 3 for 2D / 3D data

        representation_type: str = ""
        representation: dict = field(default_factory=dict)

        threshold: float = 1000
        buffer_size: int = 1024

    cfg: Config

    def configure(self) -> None:
        super().configure()

        # assert self.cfg.dim in [2, 3], f"CubicalPersistance is implemented for 2D and 3D data, found dim {self.cfg.dim}" 
        assert self.cfg.dim == 2, "CUDA cubical-ph supports only 2d for now"

        self.representation: BaseRepresentation = lmp.find(self.cfg.representation_type)(self.cfg.representation)
        self.repr_tensor_type = self.representation.tensor_type
        assert self.repr_tensor_type in ["flat", "full"]

        self.threshold = self.cfg.threshold
        self.buffer_size = self.cfg.buffer_size

    def forward(self, G_filt: Float[Tensor, "B C0 *D"], G_filt_unnorm: Float[Tensor, "B C0 *D"], G_ccl: Optional[Float[Tensor, "B C0 C1 *D"]]) -> tuple[Tensor, Tensor]:
        pers_pairs: Float[Tensor, "B C0 dim F 2"]
        pers_lengths: Int[Tensor, "B C0 dim"]
        pers_pairs, pers_lengths = cubical.cubical_persistence_v_2d_full(G_filt, self.threshold, buffer_size=self.buffer_size)

        diags_mask: Float[Tensor, "B C0 dim F"]
        diags_mask = create_diags_mask_from_pairs(pers_pairs, pers_lengths)

        B, C0, dim, F, _ = pers_pairs.shape
        
        # Calculate representations for each dimension
        pers_pairs, diags_mask = persistence_view(self.repr_tensor_type, pers_pairs, diags_mask)
        reps: Union[Float[Tensor, "B*C0*dim S"], Float[Tensor, "B *D S"]] = self.representation(pers_pairs, diags_mask)
        if self.repr_tensor_type == "flat":
            out: Float[Tensor, "B C0 S dim"] = reps.view(B, C0, dim, -1).permute(0, 1, 3, 2)
        elif self.repr_tensor_type == "full":
            out: Float[Tensor, "B *D S"] = reps

        return out, pers_pairs # type: ignore

@torch.no_grad()
def create_diags_mask_from_pairs(pers_pairs: Float[Tensor, "B C0 dim F 2"], pers_lengths: Int[Tensor, "B C dim"]) -> Tensor:
    B, C0, dim, F, _ = pers_pairs.shape
    mask = torch.arange(F, device=pers_lengths.device, dtype=pers_lengths.dtype).expand(B, C0, dim, F) < pers_lengths.unsqueeze(-1)
    return mask.to(pers_pairs.dtype)


def persistence_view(tensor_type: str, pers_pairs: Float[Tensor, "B C0 dim F 2"], diags_mask: Float[Tensor, "B C0 dim F"]) -> tuple[Tensor, Tensor]:
    B, C0, dim, F, _ = pers_pairs.shape
    if tensor_type == "flat":
        return pers_pairs.view(B*C0*dim, F, 2), diags_mask.view(B*C0*dim, F)
    elif tensor_type == "full":
        return pers_pairs, diags_mask
    else:
        raise ValueError(f"Unknown tensor type {tensor_type} in persistence_view")