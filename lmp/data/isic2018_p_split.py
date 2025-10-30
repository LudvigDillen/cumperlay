import random
import pandas as pd

from lmp.util.typing import *

from .isic2018 import ISIC2018DataModuleConfig, ISIC2018DataModule, ISIC2018Dataset

PERCENTS = [0.01, 0.05, 0.10, 0.20, 0.50]
OUTPUT_FILE = "./lmp/data/isic2018/isic2018_{split}_{ptext}.csv"
TRAIN_DATA_PERCENT_SEED = 0

def parse_dataset(dataset: ISIC2018Dataset):
    assert dataset.indices is not None
    indices: list[int] = dataset.indices

    img_paths = [dataset.data[idx].removeprefix(dataset.cfg.dataroot + "/") for idx in indices]
    labels = [dataset.targets[idx] for idx in indices]

    return pd.DataFrame({
        "mapped_index": indices,
        "img_path": img_paths,
        "label": labels
    })


for percent in PERCENTS:
    percent_text = f"{percent:.2f}".split(".")[-1]

    config = dict(
        dataroot="/data2/data/ISIC2018/task3",
        image_size=224,
        batch_size=16,
        workers=4,
        augment_extra=False,
        train_data_percent=percent,
        train_data_percent_seed=TRAIN_DATA_PERCENT_SEED,
    )

    datamodule = ISIC2018DataModule(config)
    datamodule.prepare_data()
    datamodule.setup("fit")

    train_dataset = datamodule.train_dataset
    val_dataset = datamodule.val_dataset
    
    train_df = parse_dataset(train_dataset)
    val_df = parse_dataset(val_dataset)

    train_df.to_csv(OUTPUT_FILE.format(split="train", ptext=percent_text))
    val_df.to_csv(OUTPUT_FILE.format(split="val", ptext=percent_text))


