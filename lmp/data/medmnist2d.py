import os
from dataclasses import dataclass, field
import glob
from pathlib import Path
import h5py
import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2

import lmp
from lmp import register
from lmp.util.base import Updateable
from lmp.util.config import parse_structured
from lmp.util.misc import get_rank
from lmp.util.typing import *

import medmnist
from medmnist.dataset import MedMNIST2D

@dataclass
class MedMnist2dDataModuleConfig:
    dataset: str = "breastmnist" # in DATASET_MAPPING
    dataroot: str = str("/data2/medmnist/")
    download: bool = True
    mmap_mode: Optional[str] = None

    as_rgb: bool = False
    size: Optional[int] = None # None, 28, 64, 128, 224

    train_data_percent: Optional[float] = None
    train_data_percent_seed: int = 0

    train_data_num: Optional[int] = None
    val_data_num: Optional[int] = None

    train_batch_size: int = 4
    val_batch_size: int = 4
    test_batch_size: int = 4
    train_workers: int = 8
    val_workers: int = 8
    test_workers: int = 8

    has_bifiltration_data: bool = False
    train_bifiltration_h5: str = ""

    has_betti_data: bool = False
    betti_data_npz: str = ""

    has_betti_mpb: bool = False
    betti_mpb: str = ""

    data_augment: bool = False

    augment_extra: bool = True

DATASET_MAPPING = {
    "pathmnist": medmnist.PathMNIST,
    "octmnist": medmnist.OCTMNIST,
    "pneumoniamnist": medmnist.PneumoniaMNIST,
    "chestmnist": medmnist.ChestMNIST,
    "dermamnist": medmnist.DermaMNIST,
    "retinamnist": medmnist.RetinaMNIST,
    "breastmnist": medmnist.BreastMNIST,
    "bloodmnist": medmnist.BloodMNIST,
    "tissuemnist": medmnist.TissueMNIST,
    "organamnist": medmnist.OrganAMNIST,
    "organcmnist": medmnist.OrganCMNIST,
    "organsmnist": medmnist.OrganSMNIST,
}

class MedMnist2dDataset(Dataset, Updateable):
    def __init__(self, cfg: MedMnist2dDataModuleConfig, split: str):
        super().__init__()

        self.cfg = cfg
        
        if self.cfg.data_augment:
            transform_list = [ 
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                # size should already be correct here
                # v2.Resize((self.cfg.image_size,self.cfg.image_size), antialias=True),
            ]
            if split == "train":
                transform_list.extend([
                    v2.RandomHorizontalFlip(),
                    v2.RandomVerticalFlip(),
                ])
            transform_list.extend( [
                v2.ToDtype(torch.float32, scale=True),  
                v2.Normalize(mean= [.5], std=[.5])
            ])
            if split == "train" and self.cfg.augment_extra:
                transform_list.extend([
                    v2.ColorJitter(0.02, 0.02, 0.02, 0.01),
                    v2.RandomRotation([-180, 180]),
                    v2.RandomAffine([-180, 180], translate=[0.1, 0.1],
                                    scale=[0.7, 1.3])
                ])
        else:
            transform_list = [
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5])
            ]
        data_transform = transforms.Compose(transform_list)

        
        assert cfg.dataset in DATASET_MAPPING
        dataset_cls: Type[MedMNIST2D] = DATASET_MAPPING[cfg.dataset]
        self.dataset: MedMNIST2D = dataset_cls(
            split=split,
            transform=data_transform,
            root=self.cfg.dataroot,
            download=self.cfg.download,
            as_rgb=self.cfg.as_rgb,
            size=self.cfg.size,
            mmap_mode=self.cfg.mmap_mode
        )

        self.bifilt_data = None
        if self.cfg.has_bifiltration_data and split == "train":
            self.bifilt_data = h5py.File(self.cfg.train_bifiltration_h5, 'r')
        
        self.betti_data = None
        if self.cfg.has_betti_data:
            with np.load(self.cfg.betti_data_npz.format(split=split)) as data:
                self.betti_data = data["betti"].transpose(0, 3, 1, 2).astype(np.float32)

        if self.cfg.has_betti_mpb:
            assert not self.cfg.has_betti_data and self.cfg.betti_mpb != ""
            with np.load(self.cfg.betti_mpb.format(split=split)) as data:
                self.betti_data = data["betti"].astype(np.float32)

        self.indices = None
        is_trainval = split in ["train", "val"]
        if is_trainval and self.cfg.train_data_percent is not None:
            assert 0.001 <= self.cfg.train_data_percent <= 0.999
            indices = list(range(len(self.dataset)))
            cutoff_point = int(self.cfg.train_data_percent * len(indices))

            rng = random.Random(self.cfg.train_data_percent_seed)
            rng.shuffle(indices)

            self.indices = indices[:cutoff_point]
        elif is_trainval and self.cfg.train_data_num is not None:
            num = self.cfg.train_data_num
            if split == "val":
                assert self.cfg.val_data_num is not None
                num = self.cfg.val_data_num
            indices = list(range(len(self.dataset)))
            cutoff_point = min(num, len(self.dataset))

            rng = random.Random(self.cfg.train_data_percent_seed)
            rng.shuffle(indices)

            self.indices = indices[:cutoff_point]

    def __len__(self):
        return len(self.dataset) if self.indices is None else len(self.indices)

    def __getitem__(self, index):
        if self.indices is not None:
            orig_index = index
            index = self.indices[index]
            
        img, target = self.dataset[index]
       
        data = {
            "input": img,
            'target': target.astype(np.int64)
        }

        if self.bifilt_data is not None:
            bifilt = self.bifilt_data[f"result_{index}"][:].astype(np.float32)
            bifilt = 1.0 - np.flip(np.swapaxes(bifilt, 0, 1), axis=0)
            data['bifilt'] = bifilt

        if self.betti_data is not None:
            betti_data = self.betti_data[index]
            data["betti"] = betti_data

        return data

@register("medmnist2d-datamodule")
class MedMnist2dDataModule(pl.LightningDataModule):
    cfg: MedMnist2dDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(MedMnist2dDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = MedMnist2dDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = MedMnist2dDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = MedMnist2dDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, shuffle=False, num_workers=0) -> DataLoader:
        return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, 
            batch_size=self.cfg.train_batch_size,
            shuffle=True,
            num_workers=self.cfg.train_workers
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, 
            batch_size=self.cfg.val_batch_size,
            num_workers=self.cfg.val_workers
        )

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, 
            batch_size=self.cfg.test_batch_size,
            num_workers=self.cfg.test_workers
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, 
            batch_size=self.cfg.test_batch_size,
            num_workers=self.cfg.test_workers
        )
