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
class ISIC2018DataModuleConfig:
    dataroot: str = str("/data2/ISIC2018/task3")

    image_size: int = 128 # Target image size

    has_bifiltration_data: bool = False
    train_bifiltration_h5: str = ""

    has_betti_data: bool = False
    betti_data_npz: str = ""

    has_betti_mpb: bool = False
    betti_mpb: str = ""
    betti_path_indexed: bool = False

    batch_size: int = 256

    workers: int = 8

    train_data_percent: Optional[float] = None
    train_data_percent_seed: int = 0

    augment_extra: bool = True
    binary: bool = False

# TODO: create all input and img files
def get_folder_names(isic_version):
    if isic_version=="ISIC2018":
        folder_names = {
            "train": ("ISIC2018_Task3_Training_Input","ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv"),
            "val": ("ISIC2018_Task3_Validation_Input","ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv"),
            "test": ("ISIC2018_Task3_Test_Input","ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv")
        }
    elif isic_version=="ISIC2024":
        folder_names = {
            "train_2018": ("TDA/Training_Input/ISIC2018_Task3_Training_Input","TDA/Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv"),
            "train_2024": ("TDA/Training_Input/ISIC2024_Training_Input_Subsampled","TDA/Training_GroundTruth/ISIC2024_Training_GroundTruth_Subsampled.csv"),
            "val": ("TDA/ISIC2018_Task3_Validation_Input","TDA/ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv"),
            "test": ("TDA/Test_Input","TDA/Test_GroundTruth/Test_GroundTruth.csv")
        }
    else:
        raise ValueError(f"Unknown dataset version: {isic_version}")
    return folder_names

class ISIC2018Dataset(Dataset, Updateable):
    CLASSES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    
    def __init__(self, cfg: ISIC2018DataModuleConfig, split: str):
        super().__init__()

        self.cfg = cfg
        if "ISIC2018" in cfg.dataroot:
            isic_version = "ISIC2018"
        elif "ISIC2024" in cfg.dataroot:
            isic_version = "ISIC2024"
        else:
            raise ValueError(f"Unknown dataset folder: {cfg.dataroot}")

        folder_names = get_folder_names(isic_version)

        if split == "train" and isic_version=="ISIC2024":
            datasets_to_load = ["train_2018", "train_2024"]
        else:
            datasets_to_load = [split]
        all_data = []
        all_targets = []
        for split_load in datasets_to_load:
            data_folder, target_csv = folder_names[split_load]
            data_folder = os.path.join(self.cfg.dataroot, data_folder)

            target_data = pd.read_csv(os.path.join(self.cfg.dataroot, target_csv))
            image_names = target_data["image"].values

            data = []
            targets = []
            if isic_version=="ISIC2018" or split_load=="train_2018" or split=="val":
                class_names = pd.from_dummies(target_data[self.CLASSES]).values.squeeze(-1)
                if self.cfg.binary:
                    benign = {'NV', 'BKL', 'DF', 'VASC'}
                    malignant = {'MEL', 'BCC', 'AKIEC'}
                    self.class_map = {cls: (1 if cls in malignant else 0) for cls in self.CLASSES}
                else:
                    self.class_map = {cls: i for i, cls in enumerate(self.CLASSES)}

                for path, label in zip(image_names, class_names):
                    data.append(os.path.join(data_folder, f"{path}.jpg"))
                    targets.append(self.class_map[label])
            elif isic_version=="ISIC2024" or split_load=="train_2024":
                lookup = target_data.set_index("image")["malignant"].to_dict()
                for path in image_names:
                    data.append(os.path.join(data_folder, f"{path}.jpg"))
                    targets.append(int(lookup[path]))
            else:
                raise ValueError(f"Unknown dataset folder: {data_folder}")
            all_data.extend(data)
            all_targets.extend(targets)
        self.data = all_data
        self.targets = np.array(all_targets)

        lmp.info(f"Split {split} has {len(self.data)} images")

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
        if split == "train" and self.cfg.augment_extra:
            transforms.extend([
                v2.ColorJitter(0.02, 0.02, 0.02, 0.01),
                v2.RandomRotation([-180, 180]),
                v2.RandomAffine([-180, 180], translate=[0.1, 0.1],
                                scale=[0.7, 1.3])
            ])
        self.data_transform = v2.Compose(transforms)

        self.bifilt_data = None
        if self.cfg.has_bifiltration_data and split == "train":
            self.bifilt_data = h5py.File(self.cfg.train_bifiltration_h5, 'r')
        
        self.betti_data = None
        if self.cfg.has_betti_data:
            with np.load(self.cfg.betti_data_npz.format(split=split)) as data:
                self.betti_data = data["betti"].transpose(0, 3, 1, 2).astype(np.float32)

        if self.cfg.has_betti_mpb:
            assert not self.cfg.has_betti_data and self.cfg.betti_mpb != ""
            if not self.cfg.betti_path_indexed:
                with np.load(self.cfg.betti_mpb.format(split=split)) as data:
                    self.betti_data = data["betti"].astype(np.float32)
            else:
                pass
            
        self.indices = None
        is_trainval = split in ["train", "val"]
        if is_trainval and self.cfg.train_data_percent is not None:
            assert 0.001 <= self.cfg.train_data_percent <= 0.999
            indices = list(range(len(self.data)))
            cutoff_point = int(self.cfg.train_data_percent * len(indices))

            rng = random.Random(self.cfg.train_data_percent_seed)
            rng.shuffle(indices)

            self.indices = indices[:cutoff_point]

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
       
        data = {
            "input": img,
            'target': target_label
        }

        if self.bifilt_data is not None:
            bifilt = self.bifilt_data[f"result_{index}"][:].astype(np.float32)
            bifilt = 1.0 - np.flip(np.swapaxes(bifilt, 0, 1), axis=0)
            data['bifilt'] = bifilt

        if self.betti_data is not None:
            betti_data = self.betti_data[index]
            data["betti"] = betti_data

        return data

@register("isic2018-datamodule")
class ISIC2018DataModule(pl.LightningDataModule):
    cfg: dict[Literal["train", "val", "test"], ISIC2018DataModuleConfig]

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        conf_overrides = {
            "train": cfg.pop("train_conf", dict()),
            "val": cfg.pop("val_conf", dict()),
            "test": cfg.pop("test_conf", dict())
        }

        self.cfg = {
            key: parse_structured(ISIC2018DataModuleConfig, {**cfg, **conf_override}) 
            for key, conf_override in conf_overrides.items()
        }

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = ISIC2018Dataset(self.cfg["train"], "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = ISIC2018Dataset(self.cfg["val"], "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = ISIC2018Dataset(self.cfg["test"], "test")

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
