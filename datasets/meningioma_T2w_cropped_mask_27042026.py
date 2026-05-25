import json
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
# from batchviewer import view_batch
from .base_datamodule import BaseDataModule
from .blosc2io import Blosc2IO
from .equal_class_sampler import make_k_class_balanced_trainloader


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _pad_or_center_crop_to(t: torch.Tensor, target_shape) -> torch.Tensor:
    """
    Symmetrically zero-pad (or center-crop) a 3D volume to `target_shape`.
    Accepts t of shape (D, H, W) and returns a tensor of shape `target_shape`.
    """
    assert t.dim() == 3, f"Expected 3D tensor (D, H, W); got shape {tuple(t.shape)}"
    td, th, tw = target_shape
    d, h, w = t.shape

    def _crop(x, cur, tgt, axis):
        if cur <= tgt:
            return x
        start = (cur - tgt) // 2
        end = start + tgt
        slicer = [slice(None)] * 3
        slicer[axis] = slice(start, end)
        return x[tuple(slicer)]

    t = _crop(t, d, td, 0)
    t = _crop(t, t.shape[1], th, 1)
    t = _crop(t, t.shape[2], tw, 2)

    d, h, w = t.shape
    pad_d = td - d
    pad_h = th - h
    pad_w = tw - w
    pad = (
        pad_w // 2, pad_w - pad_w // 2,
        pad_h // 2, pad_h - pad_h // 2,
        pad_d // 2, pad_d - pad_d // 2,
    )
    if any(p != 0 for p in pad):
        t = F.pad(t, pad, mode="constant", value=0.0)

    assert tuple(t.shape) == tuple(target_shape), \
        f"Pad/crop produced shape {tuple(t.shape)}, expected {tuple(target_shape)}"
    return t


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class meningioma_T2w_mask_cropped_Data(Dataset):
    """
    T2w + tumor-mask dual-channel meningioma classification dataset.

    Image and mask are loaded, padded to target_shape, stacked into a single
    2-channel tensor (C, D, H, W), then passed through the (batchgenerators-
    style) transform as the `image` key. After transform we re-binarise the
    mask channel because intensity augmentations may have perturbed it.

    Output of __getitem__:
        img:        torch.Tensor of shape (2, D, H, W) -- ch0 = T2w, ch1 = mask
        label:      int
        subject_id: str
    """

    def __init__(self, root, split, fold, transform=None,
                 target_shape=(64, 64, 64),
                 allow_missing_mask=False):
        super().__init__()

        self.case_root = (
            Path(root)
            / "nnsslPlans_onemmiso/Dataset020_UHN_Mayo_T2w_mask/Dataset020_UHN_Mayo_T2w_mask"
        )
        label_file = Path(root) / "labels.json"
        split_file = Path(root) / "splits.json"

        with open(split_file) as f:
            splits = json.load(f)
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split}")
        self.img_files = splits[str(fold)][split]

        with open(label_file) as f:
            labels = json.load(f)
        self.labels = [labels[i] for i in self.img_files]

        self.transform = transform
        self.target_shape = tuple(target_shape)
        self.allow_missing_mask = allow_missing_mask

        if split == "train":
            self._compute_class_weights()
        else:
            self.class_weights = None
            self.focal_alpha = None
            self.ce_weights = None

    # ------------------------------------------------------------------ #
    # Class weights (unchanged)
    # ------------------------------------------------------------------ #
    def _compute_class_weights(self):
        class_counts = Counter(self.labels)
        n_samples = len(self.labels)
        n_classes = len(class_counts)

        weights = np.zeros(n_classes, dtype=np.float32)
        for cls_idx in range(n_classes):
            if cls_idx in class_counts:
                weights[cls_idx] = n_samples / (n_classes * class_counts[cls_idx])
            else:
                weights[cls_idx] = 1.0

        weights_normalized = weights * n_classes / weights.sum()

        self.class_weights = weights_normalized
        self.class_counts = dict(class_counts)
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.focal_alpha = weights_normalized.copy()
        self.ce_weights = weights_normalized.copy()
        self.imbalance_ratio = weights.max() / weights.min()

        print(f"\n{'=' * 60}")
        print(f"Class Weight Computation Summary ({self.__class__.__name__})")
        print(f"{'=' * 60}")
        print(f"Total samples: {n_samples}")
        print(f"Number of classes: {n_classes}")
        print(f"Class distribution: {dict(sorted(class_counts.items()))}")
        print(f"Balanced weights: {weights_normalized}")
        print(f"Imbalance ratio: {self.imbalance_ratio:.2f}")
        print(f"{'=' * 60}\n")

    def get_loss_weights(self):
        if not hasattr(self, "class_weights") or self.class_weights is None:
            return None
        return {
            "class_weights": self.class_weights,
            "focal_alpha": self.focal_alpha.tolist(),
            "ce_weights": torch.from_numpy(self.ce_weights).float(),
            "focal_alpha_tensor": torch.from_numpy(self.focal_alpha).float(),
            "class_counts": self.class_counts,
            "imbalance_ratio": self.imbalance_ratio,
            "n_samples": self.n_samples,
            "n_classes": self.n_classes,
        }

    # ------------------------------------------------------------------ #
    # Path helpers
    # ------------------------------------------------------------------ #
    def _img_and_mask_paths(self, idx):
        case_id = self.img_files[idx].replace(".nii.gz", "")
        case_dir = self.case_root / case_id / "ses-DEFAULT"
        img_path = case_dir / f"{case_id}.b2nd"
        mask_path = case_dir / f"{case_id}__anat.b2nd"
        return img_path, mask_path

    # ------------------------------------------------------------------ #
    # __getitem__
    # ------------------------------------------------------------------ #
    def __getitem__(self, idx):
        subject_id = self.img_files[idx]
        img_path, mask_path = self._img_and_mask_paths(idx)

        # --- Load image -------------------------------------------------
        img_arr, _ = Blosc2IO.load(img_path, mode="r")
        img_np = np.asarray(img_arr[...])
        if img_np.ndim == 4 and img_np.shape[0] == 1:
            img_np = img_np[0]
        img_t = torch.from_numpy(img_np).float()

        # --- Load mask --------------------------------------------------
        if mask_path.exists():
            mask_arr, _ = Blosc2IO.load(mask_path, mode="r")
            mask_np = np.asarray(mask_arr[...])
            if mask_np.ndim == 4 and mask_np.shape[0] == 1:
                mask_np = mask_np[0]
            mask_t = torch.from_numpy(mask_np).float()
        else:
            if not self.allow_missing_mask:
                raise FileNotFoundError(
                    f"Expected mask not found: {mask_path}. "
                    f"Pass allow_missing_mask=True to fall back to a zero mask."
                )
            print(f"[WARN] No mask for {subject_id}; using zero mask.")
            mask_t = torch.zeros(self.target_shape, dtype=torch.float32)

        # --- Pad/crop both to target_shape ------------------------------
        # Both are padded with identical symmetric offsets, so voxel
        # correspondence between image and mask is preserved.
        img_t = _pad_or_center_crop_to(img_t, self.target_shape)
        mask_t = _pad_or_center_crop_to(mask_t, self.target_shape)
        mask_t = (mask_t > 0.5).float()  # ensure clean binary before transform

        # --- Stack into (C=2, D, H, W) BEFORE transform -----------------
        # batchgenerators-style transforms expect a 4D (C, D, H, W) tensor
        # under the `image` key. Passing image+mask as a 2-channel volume
        # means spatial transforms (flip, rotate, crop) apply identically
        # to both channels, which is exactly what we want. Intensity
        # transforms will also touch the mask channel; we re-binarise after.
        img = torch.stack([img_t, mask_t], dim=0)  # (2, D, H, W)

        if self.transform is not None:
            out = self.transform(**{"image": img})
            img = out["image"]

        # --- Re-binarise mask channel ----------------------------------
        # Intensity transforms (noise, blur, gamma) may have perturbed the
        # mask. Threshold at 0.5 to recover a clean binary channel.
        if img.dim() == 4 and img.shape[0] >= 2:
            mask_channel = (img[1] > 0.5).float()
            img = torch.stack([img[0], mask_channel], dim=0)
        else:
            raise RuntimeError(
                f"Expected (2, D, H, W) after transform, got {tuple(img.shape)}"
            )

        return img, self.labels[idx], subject_id

    def __len__(self):
        return len(self.img_files)


# --------------------------------------------------------------------------- #
# DataModule
# --------------------------------------------------------------------------- #
class meningioma_T2w_mask_cropped_DataModule(BaseDataModule):
    def __init__(self, use_balanced_sampling=False, allow_missing_mask=False,
                 target_shape=(64, 64, 64), **params):
        super().__init__(**params)
        self.loss_weights = None
        self.use_balanced_sampling = use_balanced_sampling
        self.allow_missing_mask = allow_missing_mask
        self.target_shape = tuple(target_shape)

    def setup(self, stage: str):
        ds_kwargs = dict(
            root=self.data_path,
            fold=self.fold,
            target_shape=self.target_shape,
            allow_missing_mask=self.allow_missing_mask,
        )
        self.train_dataset = meningioma_T2w_mask_cropped_Data(
            split="train", transform=self.train_transforms, **ds_kwargs)
        self.val_dataset = meningioma_T2w_mask_cropped_Data(
            split="val", transform=self.test_transforms, **ds_kwargs)
        self.test_dataset = meningioma_T2w_mask_cropped_Data(
            split="test", transform=self.test_transforms, **ds_kwargs)

        self.loss_weights = self.train_dataset.get_loss_weights()

    def train_dataloader(self):
        if not self.use_balanced_sampling:
            if not self.random_batches:
                print("Using standard shuffle sampling")
                trainloader = DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    worker_init_fn=seed_worker,
                    persistent_workers=True,
                )
            else:
                print("RandomSampler with replacement is used!")
                from torch.utils.data import RandomSampler
                random_sampler = RandomSampler(
                    self.train_dataset,
                    replacement=True,
                    num_samples=len(self.train_dataset),
                )
                trainloader = DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    worker_init_fn=seed_worker,
                    persistent_workers=True,
                    sampler=random_sampler,
                )
        else:
            print("Using KClassBalancedBatchSampler.")
            trainloader = make_k_class_balanced_trainloader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                persistent_workers=True,
                drop_last=False,
            )
        return trainloader

    @property
    def class_weights(self):
        if self.loss_weights is None:
            return None
        return self.loss_weights["ce_weights"]

    def get_focal_alpha(self, as_list=True):
        if self.loss_weights is None:
            return None
        return (self.loss_weights["focal_alpha"]
                if as_list else self.loss_weights["focal_alpha_tensor"])

    def get_ce_weights(self, as_tensor=True):
        if self.loss_weights is None:
            return None
        return (self.loss_weights["ce_weights"]
                if as_tensor else self.loss_weights["class_weights"])

    def print_weight_summary(self):
        if self.loss_weights is None:
            print("No weights computed (call setup() first)")
            return

        print(f"\n{'=' * 70}")
        print(f"DataModule Weight Summary")
        print(f"{'=' * 70}")
        print(f"Dataset: {self.train_dataset.__class__.__name__}")
        print(f"Fold: {self.fold}")
        print(f"Target shape: {self.target_shape}")
        print(f"Sampling strategy: "
              f"{'KClassBalancedBatchSampler' if self.use_balanced_sampling else 'Standard shuffle'}")
        print(f"Total training samples: {self.loss_weights['n_samples']}")
        print(f"Number of classes: {self.loss_weights['n_classes']}")
        print(f"\nClass Distribution:")
        for cls, count in sorted(self.loss_weights["class_counts"].items()):
            percentage = (count / self.loss_weights["n_samples"]) * 100
            print(f"  Class {cls}: {count:3d} samples ({percentage:5.1f}%)")
        print(f"\nImbalance Ratio: {self.loss_weights['imbalance_ratio']:.2f}")

        print(f"\nRecommended Loss Function:")
        if self.loss_weights["imbalance_ratio"] >= 3.0:
            print(f"  Focal Loss (severe imbalance)")
            print(f"    - alpha={self.loss_weights['focal_alpha']}")
            print(f"    - gamma=1.5 (recommended for small datasets)")
        elif self.loss_weights["imbalance_ratio"] >= 1.5:
            print(f"  Weighted Cross-Entropy (moderate imbalance)")
            print(f"    - weight={self.loss_weights['focal_alpha']}")
        else:
            print(f"  Standard Cross-Entropy (balanced)")
        print(f"{'=' * 70}\n")


def seed_worker(worker_id):
    import random
    import numpy as np
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)