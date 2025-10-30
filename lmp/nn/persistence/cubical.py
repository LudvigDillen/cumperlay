from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lmp
import lmp.nn as lnn
import lmp.nn.functional as LF
from lmp.util.typing import *

from lmp.nn.modules.pers_utils import PersistenceInformation

from .base import BasePersistence


@lmp.register("persistence-cubical")
class CubicalPersistence(BasePersistence):
    @dataclass
    class Config(BasePersistence.Config):
        dim: int = 2 # 2 or 3 for 2D / 3D data

        representation_type: str = ""
        representation: dict = field(default_factory=dict)

        parallel: bool = False
        n_jobs: int = 8
        ccl: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()

        assert self.cfg.dim in [2, 3], f"CubicalPersistance is implemented for 2D and 3D data, found dim {self.cfg.dim}" 

        if self.cfg.ccl:
            assert self.cfg.parallel, "CCL Cubical is only implemented for the parallel implementation"
        self.do_ccl = self.cfg.ccl
        cubical_kwargs = {
            "dim": self.cfg.dim,
            "calculate_gens": False
        }
        if self.cfg.parallel:
            if self.cfg.ccl:
                self.persistence = lnn.cubical_ccl.CubicalComplexCCL(n_jobs=self.cfg.n_jobs, **cubical_kwargs)
            else:
                self.persistence = lnn.cubical.CubicalComplexParallel(n_jobs=self.cfg.n_jobs, **cubical_kwargs)
        else:
            self.persistence = lnn.cubical.CubicalComplex(**cubical_kwargs)
        self.representation = lmp.find(self.cfg.representation_type)(self.cfg.representation)

    def forward(self, G_filt: Float[Tensor, "B C0 *D"], G_filt_unnorm: Float[Tensor, "B C0 *D"], G_ccl: Optional[Float[Tensor, "B C0 C1 *D"]]) -> tuple[Float[Tensor, "B C0 S dim"], Any]:
        # B List -> C0 List -> dim (homology dimensions) List -> Persistance Information
        if self.do_ccl:
            persistence_info: list[list[list[lnn.cubical.PersistenceInformation]]] = self.persistence(G_filt, G_filt_unnorm, G_ccl)
        else:
            persistence_info: list[list[list[lnn.cubical.PersistenceInformation]]] = self.persistence(G_filt)
        
        # Combine persistance_info into a B C0 D tensor, with F (padded with mask) birth/death times
        persistence_diags: Float[Tensor, "B C0 dim F 2"] 
        diags_mask: Float[Tensor, "B C0 dim F"] 
        persistence_diags, diags_mask = pack_persistence_lll_diags_to_padded(persistence_info, pad_value=0.0)
        B, C0, dim, F, _ = persistence_diags.shape
        
        # Calculate representations for each dimension
        reps: Float[Tensor, "B*C0*dim S"] = self.representation(persistence_diags.view(B*C0*dim, F, 2), diags_mask.view(B*C0*dim, F))
        out: Float[Tensor, "B C0 S dim"] = reps.view(B, C0, dim, -1).permute(0, 1, 3, 2)

        return out, persistence_info

def pack_persistence_lll_diags_to_padded(
    pers_infos: list[list[list[lnn.cubical.PersistenceInformation]]],
    pad_value: float = 0.0
) -> tuple[Float[Tensor, "B C D N 2"], Float[Tensor, "B C D N"]]:
    max_length = 0
    tensors = []
    for batch_elem in pers_infos: # for in batches
        channel_tensors = []
        for channel in batch_elem: # for in channels
            dimension_tensors = []
            for pers_info in channel: # for in dimensions
                max_length = max(max_length, len(pers_info.diagram))
                tensor_rep = persistence_info_diag_to_tensor(pers_info)
                dimension_tensors.append(tensor_rep)
            channel_tensors.append(dimension_tensors)
        tensors.append(channel_tensors)
    N = max_length

    def _pad_tensors(tensors, N, value=torch.nan):
        return list(
            map(
                lambda t: torch.nn.functional.pad(
                    t, (0, 0, 0, N - len(t)), mode="constant", value=value
                ),
                tensors,
            )
        )
    
    result = torch.stack( # Stack over batches
        [
            torch.stack( # Stack over channels
                [
                    torch.stack( # Stack over dimensions
                        _pad_tensors(dim_tensors, N, value=pad_value)
                    ) for dim_tensors in channel_tensors
                ]
            ) for channel_tensors in tensors
        ]
    )
    mask = torch.stack( # Stack over batches
        [
            torch.stack( # Stack over channels
                [
                    torch.stack( # Stack over dimensions
                        _pad_tensors([ # over each tensor
                            tensor.new_ones(tensor.shape[:-1] + (1,)) for tensor in dim_tensors
                        ], N, value=0.0)
                    ) for dim_tensors in channel_tensors
                ]
            ) for channel_tensors in tensors
        ]
    ).unsqueeze(-1)
    # mask = torch.any(~result.isnan(), dim=-1).to(result.dtype)
    # result_nonnan = result.nan_to_num(nan=pad_value)
    
    return result, mask


def persistence_info_diag_to_tensor(
    pers_info: PersistenceInformation
) -> Float[Tensor, "N 2"]:
    pairs = torch.as_tensor(pers_info.diagram, dtype=torch.float)
    return pairs

