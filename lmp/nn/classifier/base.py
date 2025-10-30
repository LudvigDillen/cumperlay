from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lmp
from lmp.util.base import BaseModule
from lmp.util.typing import *


class BaseClassifier(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pass

    cfg: Config

    def configure(self) -> None:
        super().configure()
