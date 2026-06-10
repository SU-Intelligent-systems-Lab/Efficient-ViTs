"""
Data pipeline for the Light-weight ViTs library (ImageNet-100).

This package provides:
- Dataset builder: ``build_imagenet100k_datasets`` (an ImageFolder tree).
- ImageNet normalization constants: ``IMAGENET_MEAN`` / ``IMAGENET_STD``.
- Transform pipelines (ImageNet train/eval) plus builder helpers.
- Loaders: a fixed-scale loader, and a multi-scale loader + ``MultiScaleSampler``
  implementing the MobileViT batch-size scaling rule.

Transforms are never attached to the datasets; the loader layer applies them so
the same base dataset can be reused at any resolution.
"""

from .datasets import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_imagenet100k_datasets,
)
from .loaders import (
    MultiScaleSampler,
    build_fixed_scale_loader,
    build_multi_scale_loader,
)
from .transforms import (
    build_eval_transforms,
    build_train_transforms,
    make_eval_transform,
    make_train_transform_builder,
)


__all__ = [
    # datasets
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "build_imagenet100k_datasets",
    # transforms
    "build_train_transforms",
    "build_eval_transforms",
    "make_train_transform_builder",
    "make_eval_transform",
    # loaders
    "MultiScaleSampler",
    "build_fixed_scale_loader",
    "build_multi_scale_loader",
]
