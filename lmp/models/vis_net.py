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

@lmp.register("vis-net-classifier")
class PersVisNetClassifier(BaseModel):
    @dataclass
    class Config(BaseModel.Config):
        network_type: str = ""
        network: dict = field(default_factory=dict)

    cfg: Config
    
    def configure(self) -> None:
        super().configure()

        self.network = lmp.find(self.cfg.network_type)(self.cfg.network)

    def forward(self, x: Float[Tensor, "B Cin *D"], **kwargs) -> Float[Tensor, "B Dout"]:
        pred = self.network(x)

        return pred, dict()

@lmp.register("vis-net-segmentation")
class VisNetSegmentation(BaseModel):
    @dataclass
    class Config(BaseModel.Config):
        network_type: str = ""
        network: dict = field(default_factory=dict)

        classifier_type: str = ""
        classifier: dict = field(default_factory=dict)

        aux_classifier_type: Optional[str] = None
        aux_classifier: dict = field(default_factory= dict)

    cfg: Config
    
    def configure(self) -> None:
        super().configure()

        self.network = lmp.find(self.cfg.network_type)(self.cfg.network)
        self.classifier = lmp.find(self.cfg.classifier_type)(self.cfg.classifier)
        self.aux_classifier = None
        if self.cfg.aux_classifier_type is not None:
            self.aux_classifier = lmp.find(self.cfg.aux_classifier_type)(self.cfg.aux_classifier)

    def forward(self, x: Float[Tensor, "B Cin *D"], **kwargs) -> Tuple[Float[Tensor, "B Dout"], dict[str, Any]]:
        input_shape = x.shape[-2:]
        features = self.network(x)

        result = dict(**features)
        x = features["out"]
        x = self.classifier(x)
        pred = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        
        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            aux_pred = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            result["aux"] = aux_pred

        return pred, result
    
