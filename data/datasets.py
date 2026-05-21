"""
CIFAR-100 dataset utilities.

This module provides:
- CIFAR100_MEAN: per-channel CIFAR-100 training-set mean
- CIFAR100_STD: per-channel CIFAR-100 training-set standard deviation
- CIFAR100_NUM_CLASSES: number of CIFAR-100 classes
- build_cifar100_datasets: create CIFAR-100 train and test datasets

Example
-------
>>> from data.datasets import build_cifar100_datasets
>>>
>>> train_dataset, test_dataset = build_cifar100_datasets(root="./cifar100")
>>> len(train_dataset), len(test_dataset)
(50000, 10000)
"""

from __future__ import annotations

from pathlib import Path

from torchvision.datasets import CIFAR100


__all__ = [
    "CIFAR100_MEAN",
    "CIFAR100_STD",
    "CIFAR100_NUM_CLASSES",
    "build_cifar100_datasets",
]


CIFAR100_MEAN: tuple[float, float, float] = (0.5071, 0.4865, 0.4409)
CIFAR100_STD: tuple[float, float, float] = (0.2673, 0.2564, 0.2762)
CIFAR100_NUM_CLASSES: int = 100


def build_cifar100_datasets(
    root: str | Path,
    download: bool = True,
) -> tuple[CIFAR100, CIFAR100]:
    """
    Download (if needed) and return the CIFAR-100 train and test datasets.

    Parameters
    ----------
    root:
        Directory under which CIFAR-100 will be stored. Created if missing.

    download:
        If True, download CIFAR-100 to root when it is not already present.

    Returns
    -------
    train_dataset:
        The 50,000-image CIFAR-100 training set.

    test_dataset:
        The 10,000-image CIFAR-100 test set.

    Notes
    -----
    No transform is attached to either dataset. Each __getitem__ call returns
    a (PIL.Image, int) pair. The loader layer is responsible for converting
    images to tensors and applying any resize / augmentation.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    train_dataset = CIFAR100(
        root=str(root),
        train=True,
        download=download,
        transform=None,
    )

    test_dataset = CIFAR100(
        root=str(root),
        train=False,
        download=download,
        transform=None,
    )

    return train_dataset, test_dataset
