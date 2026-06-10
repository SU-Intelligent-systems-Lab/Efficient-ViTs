"""
Dataset builders for the Light-weight ViTs library.

The benchmark dataset is **ImageNet-100** (a 100-class subset of ImageNet),
treated as an ``ImageFolder`` tree. This module provides the dataset builder and
the ImageNet normalization constants.

Design: transforms are never attached to the dataset
----------------------------------------------------
The builder returns datasets whose ``__getitem__`` yields a
``(PIL.Image, int)`` pair with **no** transform applied (``transform=None``).
The loader layer (:mod:`data.loaders`) applies the transform inside a thin
wrapper, so the *same* base dataset can be reused at any spatial resolution --
which is what the multi-scale sampler needs.

ImageNet-100 layout
-------------------
``ImageFolder`` expects one sub-directory per class under ``train/`` and
``val/``::

    imagenet100/
      train/
        n01440764/  *.JPEG
        n01443537/  *.JPEG
        ...
      val/
        n01440764/  *.JPEG
        ...

The class set is inferred from the (sorted) sub-directory names, so ``train``
and ``val`` must use identical class folders. ``ImageFolder`` loads each image
as an RGB ``PIL.Image`` and assigns an integer label from the class order.

Normalization
-------------
For an input tensor ``x`` in ``[0, 1]`` (after ``ToTensor``), normalization is
``x' = (x - mean) / std`` applied per channel, using the standard ImageNet
statistics.

Example
-------
>>> from data.datasets import build_imagenet100k_datasets
>>> train, val = build_imagenet100k_datasets("./imagenet100")
>>> num_classes = len(train.classes)
"""

from __future__ import annotations

from pathlib import Path

from torchvision.datasets import ImageFolder


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "build_imagenet100k_datasets",
]


# Standard ImageNet normalization (used for ImageNet-100 / any ImageFolder).
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def build_imagenet100k_datasets(
    root: str | Path,
    train_dir: str = "train",
    val_dir: str = "val",
) -> tuple[ImageFolder, ImageFolder]:
    """
    Return the ImageNet-100 train and validation datasets (``ImageFolder``).

    The datasets are returned with ``transform=None`` so the loader layer can
    apply resolution-specific transforms.

    Parameters
    ----------
    root:
        Root directory containing ``train_dir`` and ``val_dir`` sub-directories.

    train_dir:
        Name of the training split sub-directory. Default ``"train"``.

    val_dir:
        Name of the validation split sub-directory. Default ``"val"``.

    Returns
    -------
    train_dataset, val_dataset:
        ``ImageFolder`` datasets for the two splits. ``train_dataset.classes``
        holds the sorted class names; ``len(train_dataset.classes)`` is the
        class count.

    Raises
    ------
    FileNotFoundError
        If ``root``, the train split, or the val split does not exist, with a
        message describing the expected layout.
    """
    root = Path(root)
    train_path = root / train_dir
    val_path = root / val_dir

    if not root.exists():
        raise FileNotFoundError(
            f"ImageNet-100 root {root!r} does not exist. Expected an ImageFolder "
            f"layout: {root}/{train_dir}/<class>/*.JPEG and {root}/{val_dir}/<class>/*.JPEG."
        )
    for split_path, split_name in ((train_path, train_dir), (val_path, val_dir)):
        if not split_path.exists():
            raise FileNotFoundError(
                f"ImageNet-100 {split_name} split not found at {split_path!r}. "
                f"Expected {split_path}/<class_name>/*.JPEG."
            )

    # transform=None: images are returned as raw RGB PIL objects; the loader
    # layer attaches the resolution-specific transform.
    train_dataset = ImageFolder(str(train_path), transform=None)
    val_dataset = ImageFolder(str(val_path), transform=None)
    return train_dataset, val_dataset
