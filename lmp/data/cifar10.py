import os
from dataclasses import dataclass, field
import glob
from pathlib import Path
import h5py

import numpy as np
import pandas as pd
from skimage import io

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset, random_split
import torchvision.transforms.v2 as v2
from torchvision.datasets import CIFAR10

import lmp
from lmp import register
from lmp.util.base import Updateable
from lmp.util.config import parse_structured
from lmp.util.misc import get_rank
from lmp.util.typing import *


@dataclass
class CIFAR10DataModuleConfig:
    dataroot: str = str("/data2/CIFAR10")

    image_size: int = 128 # Target image size

    batch_size: int = 256

    workers: int = 8

    seed: int = 42

    train_percent: float = 0.9


class CIFAR10Dataset(Dataset, Updateable):
    def __init__(self, cfg: CIFAR10DataModuleConfig, split: str, indices: List[int]):
        super().__init__()

        self.cfg = cfg

        transforms = [ 
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize((self.cfg.image_size,self.cfg.image_size), antialias=True),
        ]
        if split == "train":
            transforms.extend([
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
            ])
        transforms.extend( [
            v2.ToDtype(torch.float32, scale=True),  
            v2.Normalize(mean= [0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
        if split == "train":
            transforms.extend([
                v2.ColorJitter(0.02, 0.02, 0.02, 0.01),
                v2.RandomRotation([-180, 180]),
                v2.RandomAffine([-180, 180], translate=[0.1, 0.1],
                                scale=[0.7, 1.3])
            ])
        self.data_transform = v2.Compose(transforms)

        train_split = split != "test"
        self.dataset = CIFAR10(self.cfg.dataroot, train=train_split, transform=self.data_transform, download=False)
        self.indices = indices
    
        lmp.info(f"Split {split} has {indices} / {len(self.dataset)} images")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        img, target = self.dataset[self.indices[index]]
       
        data = {
            "input": img,
            'target': target
        }

        return data

@register("cifar10-datamodule")
class CIFAR10DataModule(pl.LightningDataModule):
    cfg: dict[Literal["train", "val", "test"], CIFAR10DataModuleConfig]

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        conf_overrides = {
            "train": cfg.pop("train_conf", dict()),
            "val": cfg.pop("val_conf", dict()),
            "test": cfg.pop("test_conf", dict())
        }

        self.cfg = {
            key: parse_structured(CIFAR10DataModuleConfig, {**cfg, **conf_override}) 
            for key, conf_override in conf_overrides.items()
        }

    def setup(self, stage=None) -> None:
        if stage in [None, "fit", "validate"]:
            train_len = len(CIFAR10(self.cfg["train"].dataroot, train=True, transform=None, download=False))
            indices = list(range(train_len))
            percent = self.cfg["train"].train_percent
            train_indices, val_indices = random_split(indices, [percent, 1-percent], generator=torch.Generator().manual_seed(self.cfg["train"].seed))

            self.train_dataset = CIFAR10Dataset(self.cfg["train"], "train", train_indices)
            self.val_dataset = CIFAR10Dataset(self.cfg["val"], "val", val_indices)
        if stage in [None, "test", "predict"]:
            test_len = len(CIFAR10(self.cfg["test"].dataroot, train=False, transform=None, download=False))
            indices = list(range(test_len))

            self.test_dataset = CIFAR10Dataset(self.cfg["test"], "test", indices)

    def prepare_data(self):
        CIFAR10(self.cfg["train"].dataroot, train=True, transform=None, download=True)
        CIFAR10(self.cfg["val"].dataroot, train=True, transform=None, download=True)
        CIFAR10(self.cfg["test"].dataroot, train=False, transform=None, download=True)

    def general_loader(self, dataset, batch_size, shuffle=False, num_workers=0) -> DataLoader:
        return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, 
            batch_size=self.cfg["train"].batch_size,
            shuffle=True,
            num_workers=self.cfg["train"].workers
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, 
            batch_size=self.cfg["val"].batch_size,
            num_workers=self.cfg["val"].workers
        )

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, 
            batch_size=self.cfg["test"].batch_size,
            num_workers=self.cfg["test"].workers
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, 
            batch_size=self.cfg["test"].batch_size,
            num_workers=self.cfg["test"].workers
        )
