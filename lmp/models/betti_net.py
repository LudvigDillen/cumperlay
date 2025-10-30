from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lmp
import lmp.nn as lnn
import lmp.nn.functional as LF
from lmp.util.typing import *

from .base import BaseModel

@lmp.register("betti-net-classifier")
class BettiNetClassifier(BaseModel):
    @dataclass
    class Config(BaseModel.Config):
        network_type: str = ""
        network: dict = field(default_factory=dict)

        betti_network_type: str = ""
        betti_network: dict = field(default_factory=dict)

        classifier_type: str = ""
        classifier: dict = field(default_factory=dict)

        betti_upsample_to: Optional[int] = None
        betti_features_aux: bool = False
 
    cfg: Config
    
    def configure(self) -> None:
        super().configure()

        self.network = lmp.find(self.cfg.network_type)(self.cfg.network)
        self.betti_network = lmp.find(self.cfg.betti_network_type)(self.cfg.betti_network)
        self.classifier = lmp.find(self.cfg.classifier_type)(self.cfg.classifier)

        if self.cfg.betti_upsample_to is not None:
            self.upsample = nn.Upsample(self.cfg.betti_upsample_to, mode="bilinear")
        else:
            self.upsample = None

    def forward(self, x: Float[Tensor, "B Cin *D"], betti: Float[Tensor, "B Cb *Db"]) -> tuple[Float[Tensor, "B Dout"], dict[str, Tensor]]:
        if self.upsample is not None:
            betti = self.upsample(betti)
        betti_features = self.betti_network(betti)

        features = self.network(x, style=betti_features)

        flat_aux = None
        if self.cfg.betti_features_aux:
            flat_aux = betti_features
        pred = self.classifier(features, None, flat_aux)
        
        return pred, dict()