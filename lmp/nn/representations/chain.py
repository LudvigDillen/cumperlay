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


@lmp.register("chain-representation")
class ChainRepresentation(BaseRepresentation):
    @dataclass
    class Config(BaseRepresentation.Config):
        representations: list[dict] = field(default_factory=list)

    cfg: Config

    def configure(self) -> None:
        super().configure()

        representations = self.cfg.representations
        assert len(representations) > 0
        module_list = []
        for repr in representations:
            assert "type" in repr and "config" in repr
            repr_module = lmp.find(repr["type"])(repr["config"])
            module_list.append(repr_module)
        self.repr_list = nn.ModuleList(module_list)

    def forward(self, pers_diags: Float[Tensor, "B F 2"], diags_mask: Float[Tensor, "B F"]) -> Float[Tensor, "B S"]:
        outs = []
        for module in self.repr_list:
            outs.append(module(pers_diags, diags_mask))
        return torch.cat(outs, dim=1)

    @property
    def tensor_type(self) -> str:
        return "flat"
