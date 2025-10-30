import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import lmp
from lmp.util.typing import *

from .surrogate import surrogate_atan


class BaseSurrogateGradThresholding(nn.Module):
    surrogate_func: Callable[[Float[Tensor, "*D"]], Float[Tensor, "*D"]]

    def __init__(self, surrogate_func=None):
        super().__init__()

        if surrogate_func == None:
            self.surrogate_func = surrogate_atan()
        else:
            self.surrogate_func = surrogate_func


class SurrogateGradThresholding(BaseSurrogateGradThresholding):
    def __init__(self, threshold: float = 0.5, surrogate_func=None):
        super().__init__(surrogate_func)

        self.threshold = threshold

    def forward(self, inp: Tensor) -> Tensor:
        return self.surrogate_func(inp - self.threshold)


class LearnedSurrogateGradThresholding(BaseSurrogateGradThresholding):
    threshold: Float[Tensor, "1"]

    def __init__(self, surrogate_func=None, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(surrogate_func)

        self.register_parameter("threshold", nn.Parameter(
            torch.zeros(1, **factory_kwargs)))

    def forward(self, inp: Tensor) -> Tensor:
        return self.surrogate_func(inp - self.threshold)


class MultiSurrogateGradThresholding(BaseSurrogateGradThresholding):
    thresholds: Tensor

    def __init__(self, thresholds: list[float], surrogate_func=None, expand_inp: bool = False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(surrogate_func)

        self.F = len(thresholds)
        self.expand_inp = expand_inp

        self.register_buffer("thresholds", torch.tensor(
            thresholds, **factory_kwargs).unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        # Input:
        # If expand_inp
        # x: B F L or B F C *D / B 1 L or B 1 C *D, batch dimension first
        # Otherwise
        # x: B L or B C *D, batch dimension first

        # Output: B F L or B F C *D, where F = len(thresholds)
        if self.expand_inp:
            x = x.unsqueeze(1)
        thresholds = self.thresholds.view(1, -1, *[1 for _ in x.shape[2:]])

        return self.surrogate_func(x - thresholds)


class LearnedMultiSurrogateGradThresholding(BaseSurrogateGradThresholding):
    thresholds: Float[Tensor, "1 F"]

    def __init__(self, threshold_dim: float, surrogate_func=None, expand_inp: bool = False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(surrogate_func)

        self.F = threshold_dim
        self.expand_inp = expand_inp

        self.register_parameter("thresholds", nn.Parameter(
            torch.empty((1, self.F), **factory_kwargs)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 0.01

        nn.init.normal_(self.thresholds, std=std)

    def forward(self, x: Tensor) -> Tensor:
        # Input:
        # If expand_inp
        # x: B F L or B F C *D / B 1 L or B 1 C *D, batch dimension first
        # Otherwise
        # x: B L or B C *D, batch dimension first

        # Output: B F L or B F C *D, where F = len(thresholds)
        if self.expand_inp:
            x = x.unsqueeze(1)
        thresholds = self.thresholds.view(1, -1, *[1 for _ in x.shape[2:]])

        return self.surrogate_func(x - thresholds)
