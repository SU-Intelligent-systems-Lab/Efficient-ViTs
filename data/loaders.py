"""
This module provides:
- MultiScaleSampler: variable-resolution batch sampler
- build_fixed_scale_loader: DataLoader with one fixed spatial resolution
- build_multi_scale_loader: DataLoader with variable spatial resolution

A fixed-scale loader follows this structure:
1. Dataset wrapper: applies one fixed transform to every image.
2. DataLoader: batches samples using a constant batch size.
3. Output: image tensors with the same spatial size in every batch.

A multiscale loader follows this structure:
1. MultiScaleSampler: samples a spatial resolution for each batch.
2. Batch-size scaling: smaller resolutions use larger batches, and larger
   resolutions use smaller batches.
3. Dataset wrapper: builds and caches the transform for the sampled size.
4. DataLoader: returns batches at the sampled resolution.

Example
-------
>>> from data import (
...     build_cifar100_datasets,
...     build_train_transforms,
...     build_eval_transforms,
...     build_fixed_scale_loader,
...     build_multi_scale_loader,
... )
>>>
>>> train_ds, test_ds = build_cifar100_datasets(root="./cifar100")
>>>
>>> train_loader = build_multi_scale_loader(
...     train_ds,
...     transform_builder=build_train_transforms,
...     scales=(160, 192, 224, 256),
...     base_size=256,
...     base_batch_size=64,
... )
>>>
>>> eval_loader = build_fixed_scale_loader(
...     test_ds,
...     transform=build_eval_transforms(size=256),
...     batch_size=128,
... )
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler


__all__ = [
    "MultiScaleSampler",
    "build_fixed_scale_loader",
    "build_multi_scale_loader",
]


TransformFn = Callable[[Any], Tensor]
TransformBuilder = Callable[[int], TransformFn]


class MultiScaleSampler(Sampler[list[tuple[int, int, int]]]):
    """
    Batch sampler that varies spatial resolution and batch size per step.

    The sampler yields lists of (height, width, dataset_index) tuples. The
    dataset wrapper used by build_multi_scale_loader unpacks each tuple,
    fetches the underlying image, and applies the training transform at the
    chosen resolution.

    Given a base resolution s0 and base batch size b0, the batch size at any
    resolution s is:

        b(s) = max(1, round(b0 * s0**2 / s**2))

    Smaller resolutions yield larger batches and larger resolutions yield
    smaller batches, so the total pixel count per step is roughly constant.

    Parameters
    ----------
    dataset_length:
        Number of items in the dataset being sampled from.

    scales:
        Sequence of allowed square spatial resolutions, for example
        (160, 192, 224, 256). At each batch one resolution is sampled
        uniformly from this set.

    base_size:
        Reference resolution used to anchor the batch-size scaling.

    base_batch_size:
        Batch size used when the sampled resolution equals base_size.

    shuffle:
        If True, dataset indices are shuffled at the start of each epoch.

    seed:
        Base seed combined with the epoch number to derive the per-epoch
        RNG state.

    drop_last:
        If True, the final batch is dropped when it has fewer than the
        scale-specific batch size.
    """

    def __init__(
        self,
        dataset_length: int,
        scales: Sequence[int],
        base_size: int,
        base_batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if dataset_length <= 0:
            raise ValueError("dataset_length must be a positive integer.")

        if len(scales) == 0:
            raise ValueError("scales must contain at least one resolution.")

        if any(s <= 0 for s in scales):
            raise ValueError("all scales must be positive integers.")

        if base_size <= 0:
            raise ValueError("base_size must be a positive integer.")

        if base_batch_size <= 0:
            raise ValueError("base_batch_size must be a positive integer.")

        self.dataset_length = dataset_length
        self.scales: tuple[int, ...] = tuple(scales)
        self.base_size = base_size
        self.base_batch_size = base_batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        self.batch_size_per_scale: dict[int, int] = {
            s: max(1, round(base_batch_size * (base_size**2) / (s**2)))
            for s in self.scales
        }

        self._schedule = self._build_schedule(self.epoch)

    def _build_schedule(self, epoch: int) -> list[list[tuple[int, int, int]]]:
        """
        Precompute the full per-epoch sequence of batches.

        Indices are shuffled (if shuffle=True) and then consumed in order.
        The resolution of each batch is drawn from self.scales and the batch
        size is the corresponding entry in self.batch_size_per_scale.
        """
        index_rng = random.Random(self.seed + epoch)
        scale_rng = random.Random(self.seed + epoch + 1)

        indices = list(range(self.dataset_length))
        if self.shuffle:
            index_rng.shuffle(indices)

        schedule: list[list[tuple[int, int, int]]] = []
        position = 0

        while position < len(indices):
            scale = scale_rng.choice(self.scales)
            batch_size = self.batch_size_per_scale[scale]
            batch_indices = indices[position : position + batch_size]

            if len(batch_indices) < batch_size and self.drop_last:
                break

            schedule.append([(scale, scale, idx) for idx in batch_indices])
            position += batch_size

        return schedule

    def set_epoch(self, epoch: int) -> None:
        """
        Set the current epoch and rebuild the batch schedule.

        Call this once at the start of every epoch so that shuffling and
        scale sampling advance deterministically.
        """
        if epoch < 0:
            raise ValueError("epoch must be a non-negative integer.")

        self.epoch = epoch
        self._schedule = self._build_schedule(epoch)

    def __iter__(self) -> Iterator[list[tuple[int, int, int]]]:
        """Yield the precomputed list of batches for the current epoch."""
        for batch in self._schedule:
            yield batch

    def __len__(self) -> int:
        """Return the number of batches in the current epoch."""
        return len(self._schedule)


class _FixedScaleDataset(Dataset):
    """
    Wrap a (PIL, label) dataset with a single fixed transform.

    The transform is applied lazily in __getitem__ and the result is
    (tensor, label).
    """

    def __init__(self, base: Dataset, transform: TransformFn) -> None:
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image, label = self.base[index]
        return self.transform(image), label


class _MultiScaleDataset(Dataset):
    """
    Wrap a (PIL, label) dataset for use with MultiScaleSampler.

    __getitem__ expects a (height, width, base_index) tuple produced by the
    sampler. The training transform is built lazily for each (height, width)
    pair and cached per worker.
    """

    def __init__(
        self,
        base: Dataset,
        transform_builder: TransformBuilder,
    ) -> None:
        self.base = base
        self.transform_builder = transform_builder
        self._transform_cache: dict[tuple[int, int], TransformFn] = {}

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, key: tuple[int, int, int]) -> tuple[Tensor, int]:
        height, width, base_index = key

        cache_key = (height, width)
        transform = self._transform_cache.get(cache_key)
        if transform is None:
            transform = self.transform_builder(height)
            self._transform_cache[cache_key] = transform

        image, label = self.base[base_index]
        return transform(image), label


def build_fixed_scale_loader(
    dataset: Dataset,
    transform: TransformFn,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Build a fixed-scale DataLoader.

    Every batch contains exactly batch_size samples at the spatial size
    implied by transform. Use this for evaluation and for fixed-scale
    training.

    Parameters
    ----------
    dataset:
        Dataset whose __getitem__ returns (PIL image, label).

    transform:
        Image transform that maps a PIL image to a normalized tensor at the
        fixed target size.

    batch_size:
        Constant batch size.

    shuffle:
        If True, shuffle the dataset each epoch.

    num_workers:
        Number of DataLoader worker processes.

    pin_memory:
        If True, pin host memory to speed up CPU to GPU transfers.

    drop_last:
        If True, drop the last incomplete batch.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    if num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer.")

    wrapped = _FixedScaleDataset(base=dataset, transform=transform)

    return DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


def build_multi_scale_loader(
    dataset: Dataset,
    transform_builder: TransformBuilder,
    scales: Sequence[int],
    base_size: int,
    base_batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 0,
    drop_last: bool = True,
) -> DataLoader:
    """
    Build a multiscale DataLoader following the MobileViT paper.

    Every batch is produced at a spatial resolution sampled from scales. The
    batch size at each resolution is chosen so that batch * size**2 stays
    approximately constant, anchored to base_batch_size at base_size.

    Parameters
    ----------
    dataset:
        Dataset whose __getitem__ returns (PIL image, label).

    transform_builder:
        Callable that takes a spatial size and returns a transform mapping
        a PIL image to a normalized tensor at that size.

    scales:
        Sequence of square spatial resolutions, for example
        (160, 192, 224, 256).

    base_size:
        Reference resolution used to anchor the batch-size scaling.

    base_batch_size:
        Batch size used when the sampled resolution equals base_size.

    num_workers:
        Number of DataLoader worker processes.

    pin_memory:
        If True, pin host memory to speed up CPU to GPU transfers.

    seed:
        Base seed for the sampler's per-epoch shuffling and scale sampling.

    drop_last:
        If True, drop the final batch when it is smaller than the
        scale-specific batch size.

    Notes
    -----
    Call loader.batch_sampler.set_epoch(epoch) at the start of every epoch
    to advance the sampler's RNG deterministically.
    """
    if num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer.")

    wrapped = _MultiScaleDataset(base=dataset, transform_builder=transform_builder)

    sampler = MultiScaleSampler(
        dataset_length=len(dataset),  # type: ignore[arg-type]
        scales=scales,
        base_size=base_size,
        base_batch_size=base_batch_size,
        shuffle=True,
        seed=seed,
        drop_last=drop_last,
    )

    return DataLoader(
        wrapped,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
