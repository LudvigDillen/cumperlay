import os
from dataclasses import dataclass, field
import glob
from pathlib import Path
import h5py
import random

import numpy as np
import pandas as pd
from skimage import io

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torchvision.transforms.v2 as v2

import lmp
from lmp import register
from lmp.util.base import Updateable
from lmp.util.config import parse_structured
from lmp.util.misc import get_rank
from lmp.util.typing import *


@dataclass
class DSpritesDataModuleConfig:
    dataroot: str = str("/data2/dsprites")
    target_file_name: str = "dsprites_extended_sampled_500_{split}.npz"

    image_size: int = 128 # Target image size
    
    batch_size: int = 256

    workers: int = 8
    
    class_index: int = -1


class DSpritesDataset(Dataset, Updateable):    
    def __init__(self, cfg: DSpritesDataModuleConfig, split: str):
        super().__init__()

        self.cfg = cfg

        target_file_name = self.cfg.target_file_name.format(split=split)
        npz_file_target = os.path.join(self.cfg.dataroot, target_file_name)
        
        dataset = np.load(npz_file_target)

        imgs = dataset["imgs"]
        self.imgs = np.repeat(imgs[:, :, :, np.newaxis], 3, axis=3).astype(np.float32)
        self.latents_values = dataset["latents_values"]
        self.latents_classes = dataset["latents_classes"]

        self.targets = self.latents_classes[:, self.cfg.class_index].astype(np.int64)

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
            v2.Normalize(mean= [0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        # if split == "train":
        #     transforms.extend([
        #         v2.ColorJitter(0.02, 0.02, 0.02, 0.01),
        #         v2.RandomRotation([-180, 180]),
        #         v2.RandomAffine([-180, 180], translate=[0.1, 0.1],
        #                         scale=[0.7, 1.3])
        #     ])
        self.data_transform = v2.Compose(transforms)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.imgs[index]
        img = self.data_transform(img)

        target_label = self.targets[index]
       
        data = {
            "input": img,
            'target': target_label
        }

        return data

@register("dsprites-datamodule")
class DSpritesDataModule(pl.LightningDataModule):
    cfg: dict[Literal["train", "val", "test"], DSpritesDataModuleConfig]

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        conf_overrides = {
            "train": cfg.pop("train_conf", dict()),
            "val": cfg.pop("val_conf", dict()),
            "test": cfg.pop("test_conf", dict())
        }

        self.cfg = {
            key: parse_structured(DSpritesDataModuleConfig, {**cfg, **conf_override}) 
            for key, conf_override in conf_overrides.items()
        }

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = DSpritesDataset(self.cfg["train"], "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = DSpritesDataset(self.cfg["val"], "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = DSpritesDataset(self.cfg["test"], "test")

    def prepare_data(self):
        pass

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
