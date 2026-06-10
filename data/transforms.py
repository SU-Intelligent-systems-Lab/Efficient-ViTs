"""
Image transform pipelines for training and evaluation (ImageNet style).

The library targets ImageNet-100, so a single, standard ImageNet recipe is used:

- **Training** (``build_train_transforms``): ``RandomResizedCrop`` + horizontal
  flip. ``RandomResizedCrop`` samples a random area fraction (within ``scale``)
  and aspect ratio, crops, and resizes to ``size`` -- the standard scale
  augmentation for ImageNet-style data.
- **Evaluation** (``build_eval_transforms``): resize the shorter side to
  ``size * resize_ratio`` then center-crop to ``size`` -- the conventional
  test-time pre-processing (the ``1.14`` margin = 256/224 avoids cropping border
  content).

Both pipelines accept a PIL image and return a normalized float tensor of shape
``(3, size, size)``. Normalization is ``x' = (x - mean) / std`` per channel with
the ImageNet statistics.

Builders for the loaders
------------------------
:func:`make_train_transform_builder` returns a ``size -> transform`` callable.
The multi-scale loader needs exactly such a builder (it constructs a fresh
transform per sampled resolution); the fixed-scale loader can use
``builder(size)`` once, or call :func:`build_train_transforms` directly.

Example
-------
>>> from data.transforms import build_train_transforms, make_train_transform_builder
>>>
>>> train_tf = build_train_transforms(size=224)
>>> builder = make_train_transform_builder()      # size -> transform
>>> tf_192 = builder(192)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torchvision import transforms

from .datasets import IMAGENET_MEAN, IMAGENET_STD


__all__ = [
    "build_train_transforms",
    "build_eval_transforms",
    "make_train_transform_builder",
    "make_eval_transform",
]


# A transform maps a PIL image to a normalized tensor; a builder maps a spatial
# size to such a transform.
TransformFn = Callable[[Any], Any]
TransformBuilder = Callable[[int], TransformFn]


def build_train_transforms(
    size: int,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    scale: tuple[float, float] = (0.08, 1.0),
) -> transforms.Compose:
    """
    Build the ImageNet training transform.

    Pipeline: RandomResizedCrop(size, scale) -> RandomHorizontalFlip ->
    ToTensor -> Normalize.

    Parameters
    ----------
    size:
        Output crop size. The tensor has shape (3, size, size).

    mean, std:
        Per-channel normalization statistics. Default to ImageNet stats.

    scale:
        Lower/upper bounds on the random crop area fraction. The default
        ``(0.08, 1.0)`` is the standard ImageNet setting; widen the lower bound
        (e.g. ``(0.35, 1.0)``) for gentler augmentation on shorter schedules.
    """
    if size <= 0:
        raise ValueError("size must be a positive integer.")

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=scale),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_eval_transforms(
    size: int,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    resize_ratio: float = 1.14,
) -> transforms.Compose:
    """
    Build the ImageNet evaluation transform (deterministic).

    Pipeline: Resize(round(size * resize_ratio)) -> CenterCrop(size) ->
    ToTensor -> Normalize.

    Parameters
    ----------
    size:
        Final center-crop size. The tensor has shape (3, size, size).

    mean, std:
        Per-channel normalization statistics. Default to ImageNet stats.

    resize_ratio:
        Pre-crop resize factor relative to ``size``.
    """
    if size <= 0:
        raise ValueError("size must be a positive integer.")

    resize_size = int(round(size * resize_ratio))
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def make_train_transform_builder(
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    **kwargs: Any,
) -> TransformBuilder:
    """
    Return a ``size -> training transform`` builder.

    This is the form the multi-scale loader needs: it calls the builder with a
    fresh resolution every time a new scale is sampled.

    Parameters
    ----------
    mean, std:
        Normalization statistics. Default to ImageNet stats.

    **kwargs:
        Extra options forwarded to :func:`build_train_transforms` (e.g.
        ``scale``).

    Returns
    -------
    Callable[[int], transform]
        A builder mapping a spatial size to a transform.
    """
    return lambda size: build_train_transforms(size, mean=mean, std=std, **kwargs)


def make_eval_transform(
    size: int,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    **kwargs: Any,
) -> transforms.Compose:
    """
    Build an evaluation transform at ``size`` (convenience wrapper).

    Parameters
    ----------
    size:
        Target spatial size.

    mean, std:
        Normalization statistics. Default to ImageNet stats.

    **kwargs:
        Extra options forwarded to :func:`build_eval_transforms` (e.g.
        ``resize_ratio``).
    """
    return build_eval_transforms(size, mean=mean, std=std, **kwargs)
