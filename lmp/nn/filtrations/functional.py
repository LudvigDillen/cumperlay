from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lmp
import lmp.nn as lnn
import lmp.nn.functional as LF
from lmp.util.typing import *

from .base import BaseFiltration

@torch.compile
def create_compact_single_filtration(G_bin: Float[Tensor, "B C *D"]) -> tuple[Float[Tensor, "B 1 *D"], list[Float[Tensor, "B 1 *D"]]]:
    C = G_bin.shape[1]

    G_filt = [G_bin[:, 0:1]]
    G_out = G_filt[0]
    for i in range(1, G_bin.shape[1]):
        # Take float union
        G_filt.append(union_max_2(G_filt[i-1], G_bin[:, i:i+1]))
        # Determine top level cells
        G_out = G_out + (i+1) * (G_filt[i] - G_filt[i-1])

    # 0 1 ... C top level cells
    return G_out, G_filt

# Compiling inner ops instead of the full loops is slower, however,
# doesnt require loop unrolling and possibly allows usage of larger parameters for C0/C1
# as the compile step for 20x30 for full function doesnt fit 40gb
# @torch.compile
def create_compact_bifiltration(G_bin: Float[Tensor, "B C0 C1 *D"]) -> tuple[Float[Tensor, "B C0 *D"], list[list[Float[Tensor, "B 1 *D"]]]]:
    B, C0, C1, *D = G_bin.shape

    G_filt = []
    for i in range(C0):
        G_filt.append([])

    # Add [0,0] to filtration, Filt[0, 0]
    G_bin_00 = G_bin[:, 0:1, 0]
    G_filt[0].append(G_bin_00)
    G_out = [G_bin_00]
    
    # print(C0, C1)

    # Add [i,0] to filtration, start constructing outputs of each row
    for i in range(1, C0):
        # Take float union, Filt[i, 0] <- Union(Filt[i-1, 0], Bin[i, 0]), i >= 1
        G_filt[i].append(union_max_2_c(G_filt[i-1][0], G_bin[:, i:i+1, 0]))
        G_out.append(G_filt[i][0])
    
    # print(len(G_filt), list([len(G_filt_i) for G_filt_i in G_filt]))

    # Add [0,j] to filtration, construct output of first row
    for j in range(1, C1):
        # # Take float union, Filt[0, j] <- Union(Filt[0, j-1], Bin[0, j]), j >= 1
        # G_filt[0].append(union_max_2(G_filt[0][j-1], G_bin[:, 0:1, j]))
        
        # # Determine top level cells
        # G_out[0] = G_out[0] + (j+1) * (G_filt[0][j] - G_filt[0][j-1])

        G_diff_0j = union_max_2_mx_c(G_filt[0][j-1], G_bin[:, 0:1, j])
        G_out[0] = G_out[0] + (j+1) * G_diff_0j
        G_filt[0].append(G_filt[0][j-1] + G_diff_0j)

    # print(len(G_filt), list([len(G_filt_i) for G_filt_i in G_filt]))

    # Add [i, j] to filtration
    for i in range(1, C0):
        # Construct output of each row
        for j in range(1, C1):
            # # Take float union, Filt[i, j] <- Union(Filt[i-1, j], Filt[i, j-1], Bin[i, j]), i,j >= 1
            # G_filt[i].append(union_max_3(G_filt[i-1][j], G_filt[i][j-1], G_bin[:, i:i+1, j]))

            # # Determine top level cells
            # G_out[i] = G_out[i] + (j+1) * (G_filt[i][j] - G_filt[i][j-1])

            # [i, j-1] must be the first element here, as it is subtracted from the result
            G_diff_ij = union_max_3_mx_c(G_filt[i][j-1], G_filt[i-1][j], G_bin[:, i:i+1, j])

            G_out[i] = G_out[i] + (j+1) * G_diff_ij
            G_filt[i].append(G_filt[i][j-1] + G_diff_ij)
            
    
    # print(len(G_filt), list([len(G_filt_i) for G_filt_i in G_filt]))

    # 1. ... C1 top level cells
    return torch.cat(G_out, dim=1), G_filt


@torch.jit.script
def union_max_2(x: Tensor, y: Tensor) -> Tensor:
    return x + y - x * y

@torch.jit.script
def union_max_3(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    xy = x * y
    return x + y + z - xy - y * z - x * z + xy * z

# triton.cudagraphs or fullgraph doesnt work for the following due to the storageapi
# and the functions being called multiple times
# seg fault on torch 2.0, RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run on 2.3

@torch.compile
def union_max_2_c(x: Tensor, y: Tensor) -> Tensor:
    return x + y - x * y

@torch.compile
def union_max_2_mx_c(x: Tensor, y: Tensor) -> Tensor:
    # return y - x * y # No +x
    return y * (1-x) # No +x

@torch.compile
def union_max_3_mx_c(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    # xy = x * y
    # return y + z - xy - y * z - x * z + xy * z # No +x
    return (y + z - y*z) * (1-x) # No +x

@torch.compile 
def compact_bifiltration_inner_loop(C1: int, G_filt_curr0: Float[Tensor, "B 1 *D"], G_filt_last: list[Float[Tensor, "B 1 *D"]], G_bin_row: Float[Tensor, "B 1 C1 *D"], G_out: Float[Tensor, "B 1 *D"]):
    G_filt = [G_filt_curr0]
    for j in range(1, C1):
        G_diff_ij = union_max_3_mx_c(G_filt[j-1], G_filt_last[j], G_bin_row[:, :, j])

        G_out = G_out + (j+1) * G_diff_ij
        G_filt.append(G_filt[j-1] + G_diff_ij)
    return G_out, G_filt

def create_compact_bifiltration_ic(G_bin: Float[Tensor, "B C0 C1 *D"]) -> tuple[Float[Tensor, "B C0 *D"], list[list[Float[Tensor, "B 1 *D"]]]]:
    B, C0, C1, *D = G_bin.shape

    G_filt = []
    for i in range(C0):
        G_filt.append([])

    # Add [0,0] to filtration, Filt[0, 0]
    G_bin_00 = G_bin[:, 0:1, 0]
    G_filt[0].append(G_bin_00)
    G_out = [G_bin_00]
    
    # Add [i,0] to filtration, start constructing outputs of each row
    for i in range(1, C0):
        # Take float union, Filt[i, 0] <- Union(Filt[i-1, 0], Bin[i, 0]), i >= 1
        G_filt[i].append(union_max_2_c(G_filt[i-1][0], G_bin[:, i:i+1, 0]))
        G_out.append(G_filt[i][0])
    
    # Add [0,j] to filtration, construct output of first row
    for j in range(1, C1):
        G_diff_0j = union_max_2_mx_c(G_filt[0][j-1], G_bin[:, 0:1, j])
        G_out[0] = G_out[0] + (j+1) * G_diff_0j
        G_filt[0].append(G_filt[0][j-1] + G_diff_0j)

    # Add [i, j] to filtration
    for i in range(1, C0):
        G_out_i, G_filt_i = compact_bifiltration_inner_loop(C1, G_filt[i][0], G_filt[i-1], G_bin[:, i:i+1], G_out[i])
        G_out[i] = G_out_i
        G_filt[i] = G_filt_i

    # 1. ... C1 top level cells
    return torch.cat(G_out, dim=1), G_filt


@torch.compile 
def compact_bifiltration_inner_loop_cat(C1: int, G_filt_curr0: Float[Tensor, "B 1 *D"], G_filt_last: Float[Tensor, "B C1 *D"], G_bin_row: Float[Tensor, "B 1 C1 *D"], G_out: Float[Tensor, "B 1 *D"]):
    G_filt = [G_filt_curr0]
    for j in range(1, C1):
        G_diff_ij = union_max_3_mx_c(G_filt[j-1], G_filt_last[:, j:j+1], G_bin_row[:, :, j])

        G_out = G_out + (j+1) * G_diff_ij
        G_filt.append(G_filt[j-1] + G_diff_ij)
    return G_out, torch.cat(G_filt, dim=1)

def create_compact_bifiltration_ic_cat(G_bin: Float[Tensor, "B C0 C1 *D"]) -> tuple[Float[Tensor, "B C0 *D"], Float[Tensor, "B C0 C1 *D"]]:
    B, C0, C1, *D = G_bin.shape

    G_filt = []
    for i in range(C0):
        G_filt.append([])

    # Add [0,0] to filtration, Filt[0, 0]
    G_bin_00 = G_bin[:, 0:1, 0]
    G_filt[0].append(G_bin_00)
    G_out = [G_bin_00]
    
    # Add [i,0] to filtration, start constructing outputs of each row
    for i in range(1, C0):
        # Take float union, Filt[i, 0] <- Union(Filt[i-1, 0], Bin[i, 0]), i >= 1
        G_filt[i].append(union_max_2_c(G_filt[i-1][0], G_bin[:, i:i+1, 0]))
        G_out.append(G_filt[i][0])
    
    # Add [0,j] to filtration, construct output of first row
    for j in range(1, C1):
        G_diff_0j = union_max_2_mx_c(G_filt[0][j-1], G_bin[:, 0:1, j])
        G_out[0] = G_out[0] + (j+1) * G_diff_0j
        G_filt[0].append(G_filt[0][j-1] + G_diff_0j)
    G_filt[0] = torch.cat(G_filt[0], dim=1)

    # Add [i, j] to filtration
    for i in range(1, C0):
        G_out_i, G_filt_i = compact_bifiltration_inner_loop_cat(C1, G_filt[i][0], G_filt[i-1], G_bin[:, i:i+1], G_out[i])
        G_out[i] = G_out_i
        G_filt[i] = G_filt_i

    # 1. ... C1 top level cells
    return torch.cat(G_out, dim=1), torch.stack(G_filt, dim=1)
