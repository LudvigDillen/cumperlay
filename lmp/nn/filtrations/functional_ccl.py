from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lmp
import lmp.nn as lnn
import lmp.nn.functional as LF
from lmp.util.typing import *

from .functional import union_max_2_c, union_max_2_mx_c, union_max_3_mx_c

from ..ccl import bccl2d

@torch.compile 
def compact_bifiltration_inner_loop_ccl(C1: int, G_filt_curr0: Float[Tensor, "B 1 *D"], G_filt_last: list[Float[Tensor, "B 1 *D"]], G_bin_row: Float[Tensor, "B 1 C1 *D"], G_out: Float[Tensor, "B 1 *D"]):
    G_filt = [G_filt_curr0]
    G_ccl = []
    with torch.no_grad():
        ccl = bccl2d(G_filt_curr0.squeeze(1))
        G_ccl.append(ccl)
    for j in range(1, C1):
        G_diff_ij = union_max_3_mx_c(G_filt[j-1], G_filt_last[j], G_bin_row[:, :, j])

        G_out = G_out + (j+1) * G_diff_ij
        G_filt.append(G_filt[j-1] + G_diff_ij)

        with torch.no_grad():
            ccl = bccl2d(G_diff_ij.squeeze(1))
            G_ccl.append(ccl)
    return G_out, G_filt, G_ccl

def create_compact_bifiltration_ic_ccl(G_bin: Float[Tensor, "B C0 C1 *D"]) -> tuple[Float[Tensor, "B C0 *D"], list[list[Float[Tensor, "B 1 *D"]]]]:
    B, C0, C1, *D = G_bin.shape

    G_filt = []
    G_ccl = []

    for i in range(C0):
        G_filt.append([])
        G_ccl.append([])

    # Add [0,0] to filtration, Filt[0, 0]
    G_bin_00 = G_bin[:, 0:1, 0]
    G_filt[0].append(G_bin_00)
    G_out = [G_bin_00]

    with torch.no_grad():
        ccl = bccl2d(G_bin_00.squeeze(1))
        G_ccl[0].append(ccl)
    
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

        with torch.no_grad():
            ccl = bccl2d(G_diff_0j.squeeze(1))
            G_ccl[0].append(ccl)
    with torch.no_grad():
        G_ccl[0] = torch.stack(G_ccl[0], dim=1)

    # Add [i, j] to filtration
    for i in range(1, C0):
        G_out_i, G_filt_i, G_ccl_i = compact_bifiltration_inner_loop_ccl(C1, G_filt[i][0], G_filt[i-1], G_bin[:, i:i+1], G_out[i])
        G_out[i] = G_out_i
        G_filt[i] = G_filt_i
        with torch.no_grad():
            G_ccl[i] = torch.stack(G_ccl_i, dim=1)
    with torch.no_grad():
        G_ccl = torch.stack(G_ccl, dim=1)
    # 1. ... C1 top level cells
    return torch.cat(G_out, dim=1), G_filt, G_ccl

