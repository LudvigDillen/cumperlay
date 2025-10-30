"""Cubical complex calculation module."""
# Based on torch_topological at https://github.com/aidos-lab/pytorch-topological/blob/main/torch_topological/nn/cubical_complex.py


import numpy as np
import torch
from torch import nn

import gudhi

import lmp
import lmp.nn as lnn
import lmp.nn.functional as LF
from lmp.util.typing import *

from .pers_utils import PersistenceInformation

from joblib import Parallel, delayed

class CubicalComplexCCL(nn.Module):
    
    def __init__(self, superlevel=False, dim: int | None = None, calculate_gens: bool = True, n_jobs: int = 8):
        super().__init__()

        self.superlevel = superlevel
        self.dim = dim
        self.calculate_gens = calculate_gens

        self.n_jobs = n_jobs

    def forward(self, x: Tensor, x_unnorm: Tensor, ccl: Tensor):
        # Dimension was provided; this makes calculating the *effective*
        # dimension of the tensor much easier: take everything but the
        # last `self.dim` dimensions.
        if self.dim is not None:
            shape = x.shape[:-self.dim]
            dims = len(shape)

        # No dimension was provided; just use the shape provided by the
        # client.
        else:
            dims = len(x.shape) - 2

        if self.superlevel:
            x = -x
        
        # No additional dimensions present: a single image
        if dims == 0:
            B, C = 1, 1
            x = x.unsqueeze(0).unsqueeze(0)
            x_unnorm = x_unnorm.unsqueeze(0).unsqueeze(0)
            ccl = ccl.unsqueeze(0).unsqueeze(0)
        # Handle image with channels, such as a tensor of the form `(C, H, W)`
        elif dims == 1:
            B, C = 1, x.shape[0]
            x = x.unsqueeze(0)
            x_unnorm = x_unnorm.unsqueeze(0)
            ccl = ccl.unsqueeze(0)
        # Handle image with channels and batch index, such as a tensor of
        # the form `(B, C, H, W)`.
        elif dims == 2:
            B, C = x.shape[0], x.shape[1]
        else:
            raise ValueError("Unknown dimensions")
        
        max_dim = len(x.shape) - dims

        # Process tasks in parallel
        x_cpu = x.detach().flatten(start_dim=2).cpu()
        shape = x.shape[2:]

        cofaces_flat = Parallel(n_jobs=self.n_jobs)(delayed(cubical_cofaces)(x_cpu[i, j], shape) for i in range(B) for j in range(C))

        # cofaces = [[cofaces_flat[i * C + j] for j in range(C)] for i in range(B)]

        batch_list = []
        for i in range(B):
            channel_list = []
            for j in range(C):
                cofaces = cofaces_flat[i * C + j]
                x_ij = x[i, j]
                x_unnorm_ij = x_unnorm[i, j]
                ccl_ij = ccl[i, j]
                
                persistence_information = [
                    self._extract_generators_and_diagrams(
                        x_ij,
                        cofaces,
                        dim,
                        x_unnorm_ij,
                        ccl_ij
                    ) for dim in range(0, max_dim)
                ]
                channel_list.append(persistence_information)
            batch_list.append(channel_list)

        if dims == 0:
            return batch_list[0][0] # return list[PersistenceInformation]
        elif dims == 1:
            return batch_list[0] # return list[list[PersistenceInformation]]
        elif dims == 2:
            return batch_list # return list[list[list[PersistenceInformation]]]


    def _extract_generators_and_diagrams(self, x, cofaces, dim, x_unnorm, ccl):
        device = x.device

        pairs = torch.empty((0, 2), dtype=torch.long, device=device)

        try:
            regular_pairs = torch.as_tensor(
                cofaces[0][dim], dtype=torch.long, device=device
            )
            pairs = torch.cat(
                (pairs, regular_pairs)
            )
        except IndexError:
            pass

        try:
            infinite_pairs = torch.as_tensor(
                cofaces[1][dim], dtype=torch.long, device=device
            )
        except IndexError:
            infinite_pairs = None

        if infinite_pairs is not None:
            # 'Pair off' all the indices
            max_index = torch.argmax(x)
            fake_destroyers = torch.empty_like(infinite_pairs).fill_(max_index)

            infinite_pairs = torch.stack(
                (infinite_pairs, fake_destroyers), 1
            )

            pairs = torch.cat(
                (pairs, infinite_pairs)
            )

        return self._create_tensors_from_pairs(x, pairs, dim, x_unnorm, ccl)

    # Internal utility function to handle the 'heavy lifting:'
    # creates tensors from sets of persistence pairs.
    def _create_tensors_from_pairs(self, x: Tensor, pairs: Tensor, dim: int, x_unnorm: Tensor, ccl: Tensor):

        xs = x.shape
        device = x.device

        if self.calculate_gens:
            # TODO: instead of doing this with numpy/cpu
            #   update torch to a newer version and use torch.unravel_index!
            pairs_c = pairs.cpu()
            # Notice that `creators` and `destroyers` refer to pixel
            # coordinates in the image.
            creators = torch.as_tensor(
                    np.column_stack(
                        np.unravel_index(pairs_c[:, 0], xs)
                    ),
                    dtype=torch.long,
                    device=device
            )
            destroyers = torch.as_tensor(
                    np.column_stack(
                        np.unravel_index(pairs_c[:, 1], xs)
                    ),
                    dtype=torch.long,
                    device=device
            )
            gens = torch.as_tensor(torch.hstack((creators, destroyers)),device=device)
        else:
            gens = None
            
        persistence_diagram = extract_ccl_persistence_diagram(x, pairs, x_unnorm, ccl)
        
        # # TODO: Most efficient way to generate diagram again?
        # persistence_diagram = torch.stack((
        #     x.ravel()[pairs[:, 0]],
        #     x.ravel()[pairs[:, 1]]
        # ), 1)

        return PersistenceInformation(
                pairing=gens,
                diagram=persistence_diagram,
                dimension=dim
        )

def cubical_cofaces(top_cells, dimensions):
    cubical_complex = gudhi.CubicalComplex(
        dimensions=dimensions,
        top_dimensional_cells=top_cells
    )

    cubical_complex.compute_persistence() # persistence()
    cofaces = cubical_complex.cofaces_of_persistence_pairs()
    return cofaces

# @torch.compile(dynamic=True)
def extract_ccl_persistence_diagram(x: Tensor, pairs: Tensor, x_unnorm: Tensor, ccl: Tensor) -> Tensor:
    with torch.no_grad():
        shape = x_unnorm.shape
        idx_0 = torch.unravel_index(pairs[:, 0], shape)
        idx_1 = torch.unravel_index(pairs[:, 1], shape)

        # N,
        birth_val = x_unnorm[idx_0]
        death_val = x_unnorm[idx_1]

        # N, H, W
        birth_ccl_graphs = torch.index_select(ccl, dim=0, index=birth_val.to(torch.int32))
        death_ccl_graphs = torch.index_select(ccl, dim=0, index=death_val.to(torch.int32))
        
        idx_n = torch.arange(pairs.shape[0], dtype=pairs.dtype)
        birth_ccl_id = birth_ccl_graphs[(idx_n,) + idx_0]
        death_ccl_id = death_ccl_graphs[(idx_n,) + idx_1]

        # N, H, W
        birth_ccl_mask = (birth_ccl_graphs == birth_ccl_id.unsqueeze(-1).unsqueeze(-1)).to(torch.float32)
        death_ccl_mask = (death_ccl_graphs == death_ccl_id.unsqueeze(-1).unsqueeze(-1)).to(torch.float32)

        # N, 2, H, W
        persistence_mask = torch.stack((birth_ccl_mask, death_ccl_mask), dim=1)
    # TODO: Maybe instead of this get the indices where _ccl_graphs == _ccl_id and then take mean of them
    # TODO: Following doesnt support 3d right now since dim=(2, 3) instead of dim=(2, 3, 4)
    # N, 2
    persistence_diagram = (x.unsqueeze(0).unsqueeze(0) * persistence_mask).sum(dim=(2, 3)) / (persistence_mask.sum(dim=(2, 3)).clamp(min=1e-5))
    return persistence_diagram
    