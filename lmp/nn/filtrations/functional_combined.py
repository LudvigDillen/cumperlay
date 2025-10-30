from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import FunctionCtx

from kornia.filters import median_blur, gaussian_blur2d

import lmp
import lmp.nn as lnn
import lmp.nn.functional as LF
from lmp.util.typing import *

@torch.compile
def combine_compact_filtration(G_bin: Float[Tensor, "B C0 *D"]) -> Float[Tensor, "B C0 *D"]:
    B, C0, *D = G_bin.shape

    G_out = [G_bin[:, 0:1]]
    
    for i in range(1, C0):
        G_out.append(torch.minimum(G_out[-1], G_bin[:, i:i+1]))

    return torch.cat(G_out, dim=1)

class CombineCompactFiltrationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, input_: Tensor) -> Tensor:
        return combine_compact_filtration(input_)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        grad_input = grad_output.clone()
        return grad_input,


def combine_compact_filtration_ste_fn():
    def inner(x):
        return CombineCompactFiltrationSTE.apply(x)

    return inner

class CombineCompactFiltrationBMedianFilter(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, input_: Tensor, kernel_size=3) -> Tensor:
        ctx.kernel_size = kernel_size
        return combine_compact_filtration(input_)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        grad_input = median_blur(grad_output, ctx.kernel_size)
        return grad_input, None


def combine_compact_filtration_b_median_filter(kernel_size=3):
    def inner(x):
        return CombineCompactFiltrationBMedianFilter.apply(x, kernel_size)

    return inner

class CombineCompactFiltrationBGaussianFilter(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, input_: Tensor, kernel_size=3, sigma=1.0) -> Tensor:
        ctx.kernel_size = kernel_size
        ctx.sigma = sigma
        return combine_compact_filtration(input_)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        sigma = ctx.sigma
        grad_input = gaussian_blur2d(grad_output, ctx.kernel_size, (sigma, sigma))
        return grad_input, None, None


def combine_compact_filtration_b_gaussian_filter(kernel_size=3, sigma=1.0):
    def inner(x):
        return CombineCompactFiltrationBGaussianFilter.apply(x, kernel_size, sigma)

    return inner

