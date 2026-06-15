import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from medsurvival3d.data.base_datamodule import BaseDataModule
from medsurvival3d.utils.io import Blosc2IO


def _resolve_path(root, path):
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(root) / path


def _load_split_ids(splits_path, split, fold):
    with splits_path.open("r", encoding="utf-8") as handle:
        splits_data = json.load(handle)

    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split '{split}'. Expected one of train/val/test.")

    if isinstance(splits_data, dict) and split in splits_data:
        return list(splits_data[split])

    fold_key = str(fold)
    if isinstance(splits_data, dict) and fold_key in splits_data:
        fold_splits = splits_data[fold_key]
    elif isinstance(splits_data, list):
        fold_splits = splits_data[fold]
    else:
        raise KeyError(
            f"Could not find split '{split}' directly or fold '{fold_key}' in {splits_path}"
        )

    if split in fold_splits:
        return list(fold_splits[split])
    if split == "test" and "val" in fold_splits:
        return list(fold_splits["val"])

    raise KeyError(f"Could not find split '{split}' in fold '{fold_key}' of {splits_path}")


def _parse_survival_label(label, case_id):
    if isinstance(label, dict):
        try:
            event = label["event"]
        except KeyError as exc:
            raise KeyError(
                f"Survival label for '{case_id}' must contain 'event'."
            ) from exc

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
        try:
            time_bin, event = label
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Survival label for '{case_id}' must be a dict or a two-item sequence."
            ) from exc
        continuous_time = time_bin

    return {
        "time_bin": torch.tensor(time_bin, dtype=torch.long),
        "event": torch.tensor(event, dtype=torch.float32),
        "time": torch.tensor(continuous_time, dtype=torch.float32),
    }


def _strip_known_crop_suffix(case_id):
    for suffix in ("_crop_t1c", "_crop_t2w"):
        if case_id.endswith(suffix):
            return case_id[: -len(suffix)]
    return case_id


def _pad_or_center_crop_to(t, target_shape):
    if t.dim() == 4 and t.shape[0] == 1:
        t = t[0]
    if t.dim() != 3:
        raise ValueError(f"Expected 3D volume, got shape {tuple(t.shape)}")

    td, th, tw = target_shape

    for axis, target in enumerate((td, th, tw)):
        if t.shape[axis] <= target:
            continue
        start = (t.shape[axis] - target) // 2
        end = start + target
        slicer = [slice(None)] * 3
        slicer[axis] = slice(start, end)
        t = t[tuple(slicer)]

    d, h, w = t.shape
    pad_d = td - d
    pad_h = th - h
    pad_w = tw - w
    pad = (
        pad_w // 2,
        pad_w - pad_w // 2,
        pad_h // 2,
        pad_h - pad_h // 2,
        pad_d // 2,
        pad_d - pad_d // 2,
    )
    if any(p != 0 for p in pad):
        t = F.pad(t, pad, mode="constant", value=0.0)

    return t


def _load_b2nd_tensor(path):
    arr, _ = Blosc2IO.load(str(path), mode="r")
    tensor = torch.from_numpy(np.asarray(arr[...])).float()
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    return tensor


class COCACombinedB2NDSurvivalDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        fold,
        transform=None,
        data_subdir="nnsslPlans_onemmiso/Dataset619_coca_t1c_combined/Dataset619_coca_t1c_combined",
        label_file="survival_labels.json",
        split_file="splits.json",
        image_suffix="_crop_t1c.b2nd",
        case_id_suffix="",
        mask_suffix=None,
        session_id="ses-DEFAULT",
        target_shape=None,
        allow_missing_mask=False,
    ):
        super().__init__()
        self.root = Path(root)
        self.img_dir = _resolve_path(self.root, data_subdir)
        self.labels_path = _resolve_path(self.root, label_file)
        self.splits_path = _resolve_path(self.root, split_file)
        self.image_suffix = image_suffix
        self.case_id_suffix = case_id_suffix
        self.mask_suffix = mask_suffix
        self.session_id = session_id
        self.target_shape = tuple(target_shape) if target_shape is not None else None
        self.allow_missing_mask = allow_missing_mask
        self.transform = transform

        self.img_files = _load_split_ids(self.splits_path, split, fold)

        with self.labels_path.open("r", encoding="utf-8") as handle:
            self.labels = json.load(handle)

    def _case_id(self, subject_id):
        return f"{subject_id}{self.case_id_suffix}"

    def _image_path(self, subject_id):
        case_id = self._case_id(subject_id)
        return self.img_dir / case_id / self.session_id / f"{case_id}{self.image_suffix}"

    def _mask_path(self, subject_id):
        if self.mask_suffix is None:
            return None
        case_id = self._case_id(subject_id)
        return self.img_dir / case_id / self.session_id / f"{case_id}{self.mask_suffix}"

    def _label_for(self, subject_id):
        if subject_id in self.labels:
            return self.labels[subject_id]

        case_id = self._case_id(subject_id)
        if case_id in self.labels:
            return self.labels[case_id]

        base_case_id = _strip_known_crop_suffix(subject_id)
        if base_case_id in self.labels:
            return self.labels[base_case_id]

        raise KeyError(
            f"Could not find survival label for '{subject_id}', '{case_id}', "
            f"or '{base_case_id}' in {self.labels_path}"
        )

    def __getitem__(self, idx):
        subject_id = self.img_files[idx]
        image_path = self._image_path(subject_id)
        img = _load_b2nd_tensor(image_path)

        if self.target_shape is not None:
            img = _pad_or_center_crop_to(img, self.target_shape)

        mask_path = self._mask_path(subject_id)
        if mask_path is not None:
            if mask_path.exists():
                mask = _load_b2nd_tensor(mask_path)
                if self.target_shape is not None:
                    mask = _pad_or_center_crop_to(mask, self.target_shape)
                mask = (mask > 0.5).float()
            elif self.allow_missing_mask:
                if self.target_shape is None:
                    mask = torch.zeros_like(img)
                else:
                    mask = torch.zeros(self.target_shape, dtype=torch.float32)
            else:
                raise FileNotFoundError(
                    f"Expected mask not found: {mask_path}. "
                    f"Pass allow_missing_mask=True to fall back to a zero mask."
                )
            img = torch.stack([img, mask], dim=0)
        elif img.dim() == 3:
            img = img.unsqueeze(0)

        if self.transform:
            img = self.transform(**{"image": img})["image"]

        if mask_path is not None:
            if img.dim() != 4 or img.shape[0] < 2:
                raise RuntimeError(
                    f"Expected two-channel image after transform, got {tuple(img.shape)}"
                )
            img = torch.stack([img[0], (img[1] > 0.5).float()], dim=0)

        label = _parse_survival_label(self._label_for(subject_id), subject_id)
        return img, label

    def __len__(self):
        return len(self.img_files)


class COCACombinedB2NDDataModule(BaseDataModule):
    def __init__(
        self,
        data_subdir="nnsslPlans_onemmiso/Dataset619_coca_t1c_combined/Dataset619_coca_t1c_combined",
        label_file="survival_labels.json",
        split_file="splits.json",
        image_suffix="_crop_t1c.b2nd",
        case_id_suffix="",
        mask_suffix=None,
        session_id="ses-DEFAULT",
        target_shape=None,
        allow_missing_mask=False,
        labels_json=None,
        splits_json=None,
        **params,
    ):
        super().__init__(**params)
        self.data_subdir = data_subdir
        self.label_file = labels_json if labels_json is not None else label_file
        self.split_file = splits_json if splits_json is not None else split_file
        self.image_suffix = image_suffix
        self.case_id_suffix = case_id_suffix
        self.mask_suffix = mask_suffix
        self.session_id = session_id
        self.target_shape = tuple(target_shape) if target_shape is not None else None
        self.allow_missing_mask = allow_missing_mask

    def _make_dataset(self, split, transform):
        return COCACombinedB2NDSurvivalDataset(
            self.data_path,
            split=split,
            fold=self.fold,
            transform=transform,
            data_subdir=self.data_subdir,
            label_file=self.label_file,
            split_file=self.split_file,
            image_suffix=self.image_suffix,
            case_id_suffix=self.case_id_suffix,
            mask_suffix=self.mask_suffix,
            session_id=self.session_id,
            target_shape=self.target_shape,
            allow_missing_mask=self.allow_missing_mask,
        )

    def setup(self, stage: str):
        self.train_dataset = self._make_dataset("train", self.train_transforms)
        self.val_dataset = self._make_dataset("val", self.test_transforms)
        self.test_dataset = self._make_dataset("test", self.test_transforms)


# Backwards-compatible name for configs/imports that referenced the classification dataset.
COCACombinedB2NDDataset = COCACombinedB2NDSurvivalDataset
