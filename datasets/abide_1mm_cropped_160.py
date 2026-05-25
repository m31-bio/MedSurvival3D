import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .base_datamodule import BaseDataModule
from .blosc2io import Blosc2IO


class abide_1mm_cropped_160_Data(Dataset):
    def __init__(self, root, split, fold, transform=None):
        super().__init__()
        """
        GLvsL_median_shape Dataset
        """
        self.img_dir = Path(root) / "abide_1mm_cropped_160"
        label_file = Path(root) / "abide_1mm_cropped_160/labels.json"
        split_file = Path(root) / "abide_1mm_cropped_160/splits.json"

        with open(split_file) as f:
            self.img_files = json.load(f)[fold]["train" if split == "train" else "val"]

        with open(label_file) as f:
            labels = json.load(f)
        self.labels = [labels[i] for i in self.img_files]

        self.transform = transform

    def __getitem__(self, idx):

        img1, _ = Blosc2IO.load(self.img_dir / (self.img_files[idx][:-4] + "_crop.b2nd"), mode="r")

        if self.transform:
            img = self.transform(**{"image": torch.from_numpy(img1[...])})["image"]
        else:
            img = torch.from_numpy(img1[...])

        return img, self.labels[idx]

    def __len__(self):
        return len(self.img_files)


class abide_1mm_cropped_160_DataModule(BaseDataModule):
    def __init__(self, **params):
        super(abide_1mm_cropped_160_DataModule, self).__init__(**params)

    def setup(self, stage: str):

        self.train_dataset = abide_1mm_cropped_160_Data(
            self.data_path,
            split="train",
            transform=self.train_transforms,
            fold=self.fold,
        )
        self.val_dataset = abide_1mm_cropped_160_Data(
            self.data_path,
            split="val",
            transform=self.test_transforms,
            fold=self.fold,
        )