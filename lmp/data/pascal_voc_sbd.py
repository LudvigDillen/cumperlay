import os
from dataclasses import dataclass, field
import glob
from pathlib import Path
import random

import numpy as np
import pandas as pd
from skimage import io

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset, random_split
from torchvision.datasets import VOCSegmentation, SBDataset
from . import pascalvoc_transforms as T

import lmp
from lmp import register
from lmp.util.base import Updateable
from lmp.util.config import parse_structured
from lmp.util.misc import get_rank
from lmp.util.typing import *


@dataclass
class PascalVOCSBDDataModuleConfig:
    dataroot: str = str("/data2/home/ck223/data/PascalVOC")
    dataroot_sbd: str = str("/data2/home/ck223/data/SBD")

    trainval_split: float = 0.8
    trainval_split_seed: int = 42

    base_size: int = 520
    # crop_size: int = 480
    hflip_prob: float = 0.5

    batch_size: int = 256
    workers: int = 8


class PascalVOCSBDDataset(Dataset, Updateable):
    def __init__(self, cfg: PascalVOCSBDDataModuleConfig, split: str):
        super().__init__()

        self.cfg = cfg

        self.dataroot = Path(self.cfg.dataroot)
        self.split = split

        pascal_split = split
        
        pascal_split = "val" if split == "test" else "train"
    
        transforms = []
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        if split == "train":
            # if cfg.base_size >= 480:
            #     transforms += [
            #         T.RandomResize(
            #             min_size=int(0.5 * cfg.base_size), max_size=int(2.0 * cfg.base_size), both_dim=True
            #         )
            #     ]
            # else:
            transforms += [
                T.RandomResize(
                    min_size=int(cfg.base_size), max_size=int(cfg.base_size), both_dim=True
                )
            ]
            if cfg.hflip_prob > 0:
                transforms += [T.RandomHorizontalFlip(cfg.hflip_prob)]
            # transforms += [
            #     T.RandomCrop(cfg.crop_size),
            # ]
        else:
            transforms += [
                T.RandomResize(min_size=cfg.base_size, max_size=cfg.base_size, both_dim=True), 
            ]
        transforms += [
            T.PILToTensor(),
            T.ToDtype(torch.float, scale=True),
            T.Normalize(mean=mean, std=std),
        ]
        transforms = T.Compose(transforms)

        self.dataset = VOCSegmentation(
            self.cfg.dataroot,
            year="2012",
            image_set=pascal_split,
            transforms=transforms,
            download=False
        )
        if split != "test":
            self.dataset_sb = SBDataset(
                self.cfg.dataroot_sbd,
                image_set="train_noval",
                mode="segmentation",
                transforms=transforms,
                download=False
            )
            train_len = len(self.dataset) + len(self.dataset_sb)
            indices = list(range(train_len))
            percent = self.cfg.trainval_split
            train_indices, val_indices = random_split(indices, [percent, 1-percent], generator=torch.Generator().manual_seed(self.cfg.trainval_split_seed))
            self.indices = train_indices if split == "train" else val_indices
        else:
            self.indices = None

        lmp.info(f"Split {split}, pascal_split: {pascal_split}) has {len(self)} images")

    def __len__(self):
        return len(self.dataset) if self.indices is None else len(self.indices)
    
    def __getitem__(self, index):
        if self.indices is not None:
            index = self.indices[index]
        if self.split == "test" or index < len(self.dataset):
            inp, target = self.dataset[index]
        else:
            inp, target = self.dataset_sb[index - len(self.dataset)]
        return {
            "input": inp,
            "target": target
        }


@register("pascal-voc-sbd-datamodule")
class PascalVOCSBDDataModule(pl.LightningDataModule):
    cfg: dict[Literal["train", "val", "test"], PascalVOCSBDDataModuleConfig]

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        conf_overrides = {
            "train": cfg.pop("train_conf", dict()),
            "val": cfg.pop("val_conf", dict()),
            "test": cfg.pop("test_conf", dict()),
        }

        self.cfg = {
            key: parse_structured(PascalVOCSBDDataModuleConfig, {**cfg, **conf_override})
            for key, conf_override in conf_overrides.items()
        }

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = PascalVOCSBDDataset(self.cfg["train"], "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = PascalVOCSBDDataset(self.cfg["val"], "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = PascalVOCSBDDataset(self.cfg["test"], "test")

    def prepare_data(self):
        pass

    def general_loader(
        self, dataset, batch_size, shuffle=False, num_workers=0
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg["train"].batch_size,
            shuffle=True,
            num_workers=self.cfg["train"].workers,
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset,
            batch_size=self.cfg["val"].batch_size,
            num_workers=self.cfg["val"].workers,
        )

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset,
            batch_size=self.cfg["test"].batch_size,
            num_workers=self.cfg["test"].workers,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset,
            batch_size=self.cfg["test"].batch_size,
            num_workers=self.cfg["test"].workers,
        )
