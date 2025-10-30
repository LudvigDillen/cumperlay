from dataclasses import dataclass, field
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.misc import Permute
from torchvision.models._api import WeightsEnum

import lmp
import lmp.nn as lnn
import lmp.nn.functional as LF
from lmp.util.typing import *

from .base import BaseNetwork

SWIN_KEYS = Literal["swin-b", "swin-v2-b"]
SWIN_MAP: dict[SWIN_KEYS, dict[str, any]] = {
    "swin-b": {
        "patch_size": [4, 4],
        "embed_dim": 128,
        "depths": [2, 2, 18, 2],
        "num_heads": [4, 8, 16, 32],
        "window_size": [7, 7],
        "stochastic_depth_prob": 0.5,
        "block": None,
        "norm_layer": None,
        "downsample_layer": lnn.swin.PatchMerging,
        "weights": lnn.swin.Swin_B_Weights.DEFAULT
    },
    "swin-v2-b": {
        "patch_size": [4, 4],
        "embed_dim": 128,
        "depths": [2, 2, 18, 2],
        "num_heads": [4, 8, 16, 32],
        "window_size": [8, 8],
        "stochastic_depth_prob": 0.5,
        "block": lnn.swin.SwinTransformerBlockV2,
        "norm_layer": None,
        "downsample_layer": lnn.swin.PatchMergingV2,
        "weights": lnn.swin.Swin_V2_B_Weights.DEFAULT
    },
}

@lmp.register("network-swin-v2")
class SwinV2(BaseNetwork):
    @dataclass
    class Config(BaseNetwork.Config):
        model: str = "swin-v2-b"
        pretrained: bool = False

        mlp_ratio: float = 4.0
        dropout: float = 0.0
        attention_dropout: float = 0.0

        in_ch: int = 3
        out_ch: int = 256

    cfg: Config

    def configure(self) -> None:
        super().configure()

        num_classes = self.cfg.out_ch
        self.num_classes = num_classes

        assert self.cfg.model in SWIN_MAP.keys()
        args = SWIN_MAP[self.cfg.model]

        patch_size = args["patch_size"]
        embed_dim = args["embed_dim"]
        depths = args["depths"]
        num_heads = args["num_heads"]
        window_size = args["window_size"]
        stochastic_depth_prob = args["stochastic_depth_prob"]

        block = args["block"]
        if block is None:
            block = lnn.swin.SwinTransformerBlock
        
        norm_layer = args["norm_layer"]
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        downsample_layer = args["downsample_layer"]
        weights: Optional[WeightsEnum] = args["weights"]
        if not self.cfg.pretrained:
            weights = None

        mlp_ratio = self.cfg.mlp_ratio
        dropout = self.cfg.dropout
        attention_dropout = self.cfg.attention_dropout

        layers: List[nn.Module] = []
        # split image into non-overlapping patches
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    self.cfg.in_ch, embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])
                ),
                Permute([0, 2, 3, 1]),
                norm_layer(embed_dim),
            )
        )

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, norm_layer))

        self.features = nn.Sequential(*layers)

        num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(num_features)
        self.permute = Permute([0, 3, 1, 2])  # B H W C -> B C H W
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.head = nn.Linear(num_features, 1000)
    
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if weights is not None:
            self.load_state_dict(weights.get_state_dict(progress=True))

        # replace head after loading weights
        self.head = nn.Linear(num_features, num_classes)

        self.update()

    def update(self):
        self.patch_embedding = self.features[0]
        # print(self.patch_embedding[0].weight)
        self.stage_1 = self.features[1]
        self.stage_2 = nn.Sequential(*self.features[2:4])

        self.stage_3 = nn.Sequential(*self.features[4:6])
        self.stage_4 = nn.Sequential(*self.features[6:8])

        print(f"delete features =================================")
        del self.features


    def forward(self, x: Float[Tensor, "B Cin H W"]) -> Float[Tensor, "B Cout"]:
        x = self.patch_embedding(x)  # (3, 224, 224) -> (56, 56, 128)
        x = self.stage_1(x)  # (56, 56, 128) -> (56, 56, 128)
        x = self.stage_2(x)  # (56, 56, 128) -> (28, 28, 256)
        x = self.stage_3(x)  # (28, 28, 256) -> (14, 14, 512)
        x = self.stage_4(x)  # (14, 14, 512) -> (7, 7, 1024)

        # x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)  # (7, 7, 1024) -> (1024, 7, 7)
        x = self.avgpool(x)  # (1024, 7, 7) -> (1024, 1, 1)
        x = self.flatten(x)  # (1024, 1, 1) -> (1024)
        x = self.head(x)  # (1024) -> (2)
        return x

