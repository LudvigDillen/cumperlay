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
class GlaucomaDataModuleConfig:
    dataroot: str = str("/data2/data/EyeDiseaseImageDataset")
    split_file: str = str("/data2/data/EyeDiseaseImageDataset/glaucoma_{split}.csv")

    image_size: int = 224

    batch_size: int = 256

    workers: int = 8

    train_data_percent: Optional[float] = None
    train_data_percent_seed: int = 0

    augment_extra: bool = True
    augment_extra_seperate: bool = True

    has_betti_mpb: bool = False
    betti_mpb: str = ""

class GlaucomaDataset(Dataset, Updateable):
    def __init__(self, cfg: GlaucomaDataModuleConfig, split: str):
        super().__init__()

        self.cfg = cfg

        self.dataroot = Path(self.cfg.dataroot)

        image_data = pd.read_csv(self.cfg.split_file.format(dataroot=str(self.dataroot), split=split))
        image_names = image_data["image"].values.tolist()
        labels = image_data["label"].values.astype(np.int64)

        data = [self._decode_image_path(path) for path in image_names]
        self.data = data
        self.targets = labels.reshape(labels.shape[0], 1)

        lmp.info(f"Split {split} has {len(data)} images")

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
            v2.Normalize(mean= [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.data_transform_extra = None
        if split == "train" and self.cfg.augment_extra:
            extra_t = [
                    v2.ColorJitter(0.02, 0.02, 0.02, 0.01),
                    v2.RandomRotation([-180, 180]),
                    v2.RandomAffine([-180, 180], translate=[0.1, 0.1],
                                    scale=[0.7, 1.3])
                ]
            if self.cfg.augment_extra_seperate:
                self.data_transform_extra = v2.Compose(extra_t)
            else:
                transforms.extend(extra_t)
        self.data_transform = v2.Compose(transforms)

        self.betti_data = None
        if self.cfg.has_betti_mpb:
            assert self.cfg.betti_mpb != ""
            with np.load(self.cfg.betti_mpb.format(split=split)) as data:
                self.betti_data = data["betti"].astype(np.float32)

        self.indices = None
        is_trainval = split in ["train", "val"]
        if is_trainval and self.cfg.train_data_percent is not None:
            assert 0.001 <= self.cfg.train_data_percent <= 0.999
            indices = list(range(len(self.data)))
            cutoff_point = int(self.cfg.train_data_percent * len(indices))

            rng = random.Random(self.cfg.train_data_percent_seed)
            rng.shuffle(indices)

            self.indices = indices[:cutoff_point]
    
    def _decode_image_path(self, img_path):
        # data_folder = self.dataroot / Path(img_path).parts[0]
        # assert data_folder.exists()
        # img_paths = list(data_folder.rglob("*.png"))
        # assert len(img_paths) == 1
        # return img_paths[0]
    
        img: Path = self.dataroot / img_path
        assert img.exists()
        return str(img)
    
    def __len__(self):
        return len(self.data) if self.indices is None else len(self.indices)

    def __getitem__(self, index):
        if self.indices is not None:
            orig_index = index
            index = self.indices[index]
        
        img_path = self.data[index]
        target_label = self.targets[index]
        
        img = io.imread(img_path)
        img = self.data_transform(img)
        img_no_aug = None
        if self.data_transform_extra is not None:
            img_no_aug = img
            img = self.data_transform_extra(img)

        data = {
            "input": img,
            'target': target_label
        }
        if self.data_transform_extra is not None:
            data["input_no_aug"] = img_no_aug

        if self.betti_data is not None:
            betti_data = self.betti_data[index]
            data["betti"] = betti_data

        return data

@register("glaucoma-datamodule")
class GlaucomaDataModule(pl.LightningDataModule):
    cfg: dict[Literal["train", "val", "test"], GlaucomaDataModuleConfig]

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        conf_overrides = {
            "train": cfg.pop("train_conf", dict()),
            "val": cfg.pop("val_conf", dict()),
            "test": cfg.pop("test_conf", dict())
        }

        self.cfg = {
            key: parse_structured(GlaucomaDataModuleConfig, {**cfg, **conf_override}) 
            for key, conf_override in conf_overrides.items()
        }

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = GlaucomaDataset(self.cfg["train"], "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = GlaucomaDataset(self.cfg["val"], "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = GlaucomaDataset(self.cfg["test"], "test")

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
