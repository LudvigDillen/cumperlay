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

@lmp.register("learned-persistence-classifier")
class LearnedPersistenceClassifier(BaseModel):
    @dataclass
    class Config(BaseModel.Config):
        network_type: str = ""
        network: dict = field(default_factory=dict)

        filtration_type: str = ""
        filtration: dict = field(default_factory=dict)

        persistence_type: str = ""
        persistence: dict = field(default_factory=dict)

        classifier_type: str = ""
        classifier: dict = field(default_factory=dict)

        use_aux_features: bool = False
        use_aux_mean_filt_act: bool = False

    cfg: Config
    
    def configure(self) -> None:
        super().configure()

        self.network = lmp.find(self.cfg.network_type)(self.cfg.network)
        self.filtration = lmp.find(self.cfg.filtration_type)(self.cfg.filtration)
        self.persistence = lmp.find(self.cfg.persistence_type)(self.cfg.persistence)
        self.classifier = lmp.find(self.cfg.classifier_type)(self.cfg.classifier)

        self.use_aux_features = self.cfg.use_aux_features
        self.use_aux_mean_filt_act = self.cfg.use_aux_mean_filt_act

    def forward(self, x: Float[Tensor, "B Cin *D"]) -> Float[Tensor, "B Dout"]:
        # Calculate feature vectors over 2 dimensions
        G: Float[Tensor, "B C0 C1 *D"]
        feat_latent: Float[Tensor, "B Cmid *D"]
        G, feat_latent = self.network(x)
        # Calculate compact filtration representations (grayscale, top level cells)
        G_hat_norm: Float[Tensor, "B C0 *D"]
        G_hat: Float[Tensor, "B C0 *D"]
        G_filts: list[list[Float[Tensor, "B 1 *D"]]]
        G_hat_norm, G_hat, G_filts, G_ccl = self.filtration(G)
        # Calculate persistance representation
        G_p: Float[Tensor, "B C0 S dim"]
        G_p, pers_info = self.persistence(G_hat_norm, G_hat, G_ccl)

        aux_flat = None
        if self.cfg.use_aux_mean_filt_act:
            G_filts_t = torch.cat([x for l in G_filts for x in l], dim=1)
            aux_flat = G_filts_t.mean(dim=(2, 3)) # TODO: for 3d case
        pred: Float[Tensor, "B Dout"] = self.classifier(G_p, feat_latent if self.use_aux_features else None, aux_flat)
            
        return pred, {
            "feat": G, # features
            "comp_filt": G_hat, # compact filtration as top level elementary cells
            "comp_filt_norm": G_hat_norm, # compact filtration as top level elementary cells, normalized to [0, 1]
            "filts": G_filts,
            "pers_rep": G_p, # persistence representation
            "pers_info": pers_info # persistence information
        }