"""
This module provides:
- build_train_transforms: training-time image transform pipeline
- build_eval_transforms: evaluation/inference image transform pipeline

A training transform pipeline follows this structure:
1. Resize: scales the image to the requested square spatial size.
2. RandomHorizontalFlip: flips the image left-right with probability 0.5.
3. RandomVerticalFlip: flips the image top-bottom with probability 0.5.
4. ToTensor: converts a PIL image to a float tensor in [0, 1].
5. Normalize: subtracts the channel mean and divides by the channel std.

An evaluation transform pipeline follows this structure:
1. Resize: scales the image to the requested square spatial size.
2. ToTensor: converts a PIL image to a float tensor in [0, 1].
3. Normalize: subtracts the channel mean and divides by the channel std.

Both transforms accept a PIL image and return a tensor of shape
(3, size, size). Default normalization uses CIFAR-100 statistics.

Example
-------
>>> from data.transforms import build_train_transforms, build_eval_transforms
>>>
>>> train_transform = build_train_transforms(size=256)
>>> eval_transform = build_eval_transforms(size=256)
"""

from __future__ import annotations

from torchvision import transforms

from .datasets import CIFAR100_MEAN, CIFAR100_STD


__all__ = [
    "build_train_transforms",
    "build_eval_transforms",
]


def build_train_transforms(
    size: int,
    mean: tuple[float, float, float] = CIFAR100_MEAN,
    std: tuple[float, float, float] = CIFAR100_STD,
) -> transforms.Compose:
    """
    Build the training transform pipeline at a given spatial size.

    Parameters
    ----------
    size:
        Target spatial size. The output tensor has shape (3, size, size).

    mean:
        Per-channel normalization means. Defaults to CIFAR-100 statistics.

    std:
        Per-channel normalization standard deviation. Defaults to CIFAR-100
        statistics.
    """
    if size <= 0:
        raise ValueError("size must be a positive integer.")

    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_eval_transforms(
    size: int,
    mean: tuple[float, float, float] = CIFAR100_MEAN,
    std: tuple[float, float, float] = CIFAR100_STD,
) -> transforms.Compose:
    """
    Build the evaluation transform pipeline at a fixed spatial size.

    Parameters
    ----------
    size:
        Target spatial size. The output tensor has shape (3, size, size).

    mean:
        Per-channel normalization means. Defaults to CIFAR-100 statistics.

    std:
        Per-channel normalization standard deviation. Defaults to CIFAR-100
        statistics.
    """
    if size <= 0:
        raise ValueError("size must be a positive integer.")

    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
