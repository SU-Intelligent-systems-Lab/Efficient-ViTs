"""
This package provides:
- build_cifar100_datasets: create CIFAR-100 train and test datasets
- CIFAR100_MEAN, CIFAR100_STD, CIFAR100_NUM_CLASSES: CIFAR-100 constants
- build_train_transforms: training-time image transform pipeline
- build_eval_transforms: evaluation/inference image transform pipeline
- MultiScaleSampler: batch sampler for multiscale training
- build_fixed_scale_loader: DataLoader for one fixed image resolution
- build_multi_scale_loader: DataLoader for variable-resolution training
"""

from .datasets import (
    CIFAR100_MEAN,
    CIFAR100_NUM_CLASSES,
    CIFAR100_STD,
    build_cifar100_datasets,
)
from .loaders import (
    MultiScaleSampler,
    build_fixed_scale_loader,
    build_multi_scale_loader,
)
from .transforms import (
    build_eval_transforms,
    build_train_transforms,
)


__all__ = [
    "CIFAR100_MEAN",
    "CIFAR100_STD",
    "CIFAR100_NUM_CLASSES",
    "build_cifar100_datasets",
    "build_train_transforms",
    "build_eval_transforms",
    "MultiScaleSampler",
    "build_fixed_scale_loader",
    "build_multi_scale_loader",
]
