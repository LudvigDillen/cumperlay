import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import lmp
from lmp.util.typing import *

class MultifiltRegularizationLoss(nn.Module):
    clamped: bool

    sg_curr: bool
    sg_next: bool

    def __init__(self, clamped: bool = True, sg_curr: bool = False, sg_next: bool = False) -> None:
        super().__init__()

        self.clamped = clamped
        self.sg_curr = sg_curr
        self.sg_next = sg_next
        assert not (self.sg_curr and self.sg_next)

    def forward(self, current: Tensor, next: Tensor) -> Tensor:
        if self.sg_curr:
            current = current.detach()
        if self.sg_next:
            next = next.detach()
        diff = next - current
        if self.clamped:
            diff = diff.clamp(min=0.0) # max(0, n-c)
        # sum over channels, averaged across batch and h/w
        return diff.sum(dim=1).mean()
