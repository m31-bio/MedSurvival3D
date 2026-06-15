import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from medsurvival3d.data.base_datamodule import BaseDataModule
from medsurvival3d.utils.io import Blosc2IO


def _load_split_ids(split_file, split, fold):
    with open(split_file) as f:
        splits_data = json.load(f)

    if isinstance(splits_data, list):
        fold_splits = splits_data[fold]
    else:
        fold_splits = splits_data.get(str(fold), splits_data)

    split_key = split if split in fold_splits else "val"
    return fold_splits[split_key]


class SurvivalData(Dataset):
    def __init__(
        self,
        root,
        split,
        fold,
        transform=None,
        label_file="survival_labels.json",
        image_dir="nnUNetResEncUNetLPlans_3d_fullres",
        split_file="splits_final.json",
    ):
        super().__init__()
        self.img_dir = Path(root) / image_dir

        self.img_files = _load_split_ids(
            Path(root) / split_file,
            split="train" if split == "train" else split,
            fold=fold,
        )

        with open(Path(root) / label_file) as f:
            self.labels = json.load(f)

        self.transform = transform

    def __getitem__(self, idx):
        case_id = self.img_files[idx]
        img, _ = Blosc2IO.load(self.img_dir / (case_id + ".b2nd"), mode="r")

        if self.transform:
            img = self.transform(**{"image": torch.from_numpy(img[...])})["image"]
        else:
            img = torch.from_numpy(img[...])

        label = self.labels[case_id]
        if isinstance(label, dict):
            event = label["event"]
            if "time" in label:
                continuous_time = label["time"]
            elif "time_years" in label:
                continuous_time = label["time_years"]
            elif "time_months" in label:
                continuous_time = float(label["time_months"]) / 12.0
            elif "time_bin" in label:
                continuous_time = label["time_bin"]
            else:
                raise KeyError(
                    f"Survival label for '{case_id}' must contain 'time', "
                    "'time_years', 'time_months', or legacy 'time_bin'."
                )
            time_bin = label.get("time_bin", continuous_time)
        else:
            time_bin, event = label
            continuous_time = time_bin

        return img, {
            "time_bin": torch.tensor(time_bin, dtype=torch.long),
            "event": torch.tensor(event, dtype=torch.float32),
            "time": torch.tensor(continuous_time, dtype=torch.float32),
        }

    def __len__(self):
        return len(self.img_files)


class SurvivalDataModule(BaseDataModule):
    def __init__(self, **params):
        super(SurvivalDataModule, self).__init__(**params)
        self.label_file = params.get("label_file", "survival_labels.json")
        self.image_dir = params.get(
            "image_dir", "nnUNetResEncUNetLPlans_3d_fullres"
        )
        self.split_file = params.get("split_file", "splits_final.json")

    def setup(self, stage: str):
        self.train_dataset = SurvivalData(
            self.data_path,
            split="train",
            transform=self.train_transforms,
            fold=self.fold,
            label_file=self.label_file,
            image_dir=self.image_dir,
            split_file=self.split_file,
        )
        self.val_dataset = SurvivalData(
            self.data_path,
            split="val",
            transform=self.test_transforms,
            fold=self.fold,
            label_file=self.label_file,
            image_dir=self.image_dir,
            split_file=self.split_file,
        )
        self.test_dataset = SurvivalData(
            self.data_path,
            split="test",
            transform=self.test_transforms,
            fold=self.fold,
            label_file=self.label_file,
            image_dir=self.image_dir,
            split_file=self.split_file,
        )
