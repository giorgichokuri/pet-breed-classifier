"""Data loading: Oxford-IIIT Pet (37 breeds), splits and augmentation.

The dataset ships with an official split: `trainval` (3,680 images) and `test` (3,669).
We split `trainval` into train (85%) and validation (15%), stratified by breed, so that
model selection (early stopping, choosing the best epoch) never looks at the test set.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms as T

from .config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, NUM_CLASSES, RESIZE_SIZE, SEED


def build_transforms(aug: str = "basic"):
    """Return (train_transform, eval_transform).

    aug = "none":   no augmentation (only resize + crop), used for the ablation experiment
    aug = "basic":  random crop + flip + mild color jitter
    aug = "strong": basic + TrivialAugment + RandomErasing
    """
    normalize = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    eval_tf = T.Compose([T.Resize(RESIZE_SIZE), T.CenterCrop(IMG_SIZE), *normalize])

    if aug == "none":
        train_tf = eval_tf
    elif aug == "basic":
        train_tf = T.Compose([
            T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2),
            *normalize,
        ])
    elif aug == "strong":
        train_tf = T.Compose([
            T.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.TrivialAugmentWide(),
            *normalize,
            T.RandomErasing(p=0.25),
        ])
    else:
        raise ValueError(f"unknown aug: {aug}")
    return train_tf, eval_tf


class TransformSubset(Dataset):
    """A subset of a dataset with its own transform (train and val need different ones)."""

    def __init__(self, base, indices, transform):
        self.base, self.indices, self.transform = base, list(indices), transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[self.indices[i]]
        return self.transform(img), label


def _targets(ds) -> np.ndarray:
    if hasattr(ds, "_labels"):            # OxfordIIITPet stores labels here
        return np.asarray(ds._labels)
    return np.asarray([ds[i][1] for i in range(len(ds))])


def load_datasets(data_dir: str = "data", aug: str = "basic", val_size: float = 0.15,
                  fake: bool = False):
    """Return train, val, test datasets and the list of class names.

    fake=True uses random images instead of the real dataset (for quick tests without download).
    """
    train_tf, eval_tf = build_transforms(aug)
    if fake:
        trainval = datasets.FakeData(size=148, image_size=(3, 260, 300), num_classes=NUM_CLASSES,
                                     random_offset=0)
        test = datasets.FakeData(size=74, image_size=(3, 260, 300), num_classes=NUM_CLASSES,
                                 random_offset=1000)
        classes = [f"class_{i}" for i in range(NUM_CLASSES)]
    else:
        trainval = datasets.OxfordIIITPet(data_dir, split="trainval", target_types="category", download=True)
        test = datasets.OxfordIIITPet(data_dir, split="test", target_types="category", download=True)
        classes = list(trainval.classes)

    y = _targets(trainval)
    idx = np.arange(len(trainval))
    stratify = y if not fake else None
    train_idx, val_idx = train_test_split(idx, test_size=val_size, random_state=SEED, stratify=stratify)

    train_ds = TransformSubset(trainval, train_idx, train_tf)
    val_ds = TransformSubset(trainval, val_idx, eval_tf)
    test_ds = TransformSubset(test, range(len(test)), eval_tf)
    return train_ds, val_ds, test_ds, classes


def make_loaders(train_ds, val_ds, test_ds, batch_size: int = 32, num_workers: int = 2):
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train_ds, shuffle=True, drop_last=False, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
    )
