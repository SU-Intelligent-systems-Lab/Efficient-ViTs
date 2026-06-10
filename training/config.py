"""
Training configuration for the Light-weight ViTs library.

This module provides :class:`TrainConfig`, a frozen dataclass that bundles every
hyperparameter of one training run -- model choice, dataset, optimizer and
scheduler selection, regularisation, the validation monitor for best-model
checkpointing, and the AMP / gradient-clipping / multi-scale switches.

It is a single, validated, self-documenting description of a run. The universal
training script (``scripts/train.py``) accepts the same options on the command
line; this dataclass is the canonical reference for what they mean and which
values are legal (fields are validated in ``__post_init__`` so a bad config
fails at construction, not mid-training).

Example
-------
>>> from training.config import TrainConfig
>>>
>>> config = TrainConfig(
...     model="efficientvit_m0",
...     dataset="imagenet100k",
...     epochs=100,
...     batch_size=128,
...     lr=1e-3,
...     optimizer="adamw",
...     scheduler="cosine",
...     monitor="top1",
... )
"""

from __future__ import annotations

from dataclasses import dataclass

from .optimizer import OPTIMIZERS, SCHEDULERS


__all__ = ["TrainConfig"]


# Metrics that may be used as the validation monitor for best-model selection.
_MONITORS = ("top1", "top5", "f1", "precision", "recall", "loss")


@dataclass(frozen=True)
class TrainConfig:
    """
    Hyperparameter configuration for one training run.

    Parameters
    ----------
    model:
        Registry model name (e.g. ``"mobilevit_xs"``, ``"efficientvit_m0"``,
        ``"torchvision_efficientnet_b0"``).

    dataset:
        Dataset name. Currently ``"imagenet100k"`` (an ImageFolder tree).

    data_root:
        Root directory of the dataset.

    epochs:
        Maximum number of training epochs.

    batch_size:
        Training batch size (the base batch size for multi-scale runs).

    image_size:
        Input resolution. For fixed-resolution models (EfficientViT,
        EfficientFormer) this also defines the architecture.

    num_classes:
        Number of output classes. When 0 the script infers it from the dataset.

    lr:
        Peak learning rate.

    weight_decay:
        Weight decay (excluded from norms/biases by ``build_optimizer``).

    optimizer:
        Optimizer name. One of :data:`training.optimizer.OPTIMIZERS`.

    scheduler:
        LR scheduler name. One of :data:`training.optimizer.SCHEDULERS`.

    min_lr:
        Final LR for the cosine schedule.

    warmup_epochs:
        Number of linear warmup epochs (0 disables warmup).

    label_smoothing:
        Label smoothing for ``nn.CrossEntropyLoss``. Must be in [0, 1).

    classifier_dropout, drop_path:
        Regularisation passed to models that support them (e.g. MobileViT).
        Must be in [0, 1].

    monitor:
        Validation metric used to select the best checkpoint. One of
        ``top1``, ``top5``, ``f1``, ``precision``, ``recall``, ``loss``.

    amp:
        Enable automatic mixed precision.

    grad_clip_norm:
        Max global gradient norm for clipping (0 / None disables).

    multi_scale:
        Use the multi-scale sampler (only for dynamic-input models).

    scales:
        Resolutions sampled in multi-scale training.

    output:
        Path for the best-model checkpoint (``best.pt``).

    device:
        Device string ("cuda"/"cpu"). None -> auto.

    num_workers:
        DataLoader worker processes.

    early_stopping_patience:
        Stop after this many epochs without monitor improvement (0 disables).

    seed:
        Random seed.
    """

    model: str = "mobilevit_xs"
    dataset: str = "imagenet100k"
    data_root: str = "./imagenet100"

    epochs: int = 100
    batch_size: int = 128
    image_size: int = 224
    num_classes: int = 0  # 0 -> infer from dataset

    lr: float = 1e-3
    weight_decay: float = 0.05
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    min_lr: float = 1e-5
    warmup_epochs: int = 0

    label_smoothing: float = 0.1
    classifier_dropout: float = 0.0
    drop_path: float = 0.0

    monitor: str = "top1"
    amp: bool = False
    grad_clip_norm: float | None = None
    multi_scale: bool = False
    scales: tuple[int, ...] = (160, 192, 224, 256)

    output: str = "best.pt"
    device: str | None = None
    num_workers: int = 4
    early_stopping_patience: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")
        if self.num_classes < 0:
            raise ValueError("num_classes cannot be negative (0 means infer).")

        if self.lr <= 0:
            raise ValueError("lr must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")
        if self.optimizer.lower() not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}.")
        if self.scheduler.lower() not in SCHEDULERS:
            raise ValueError(f"scheduler must be one of {SCHEDULERS}.")
        if self.min_lr < 0:
            raise ValueError("min_lr cannot be negative.")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs cannot be negative.")

        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1).")
        if not 0.0 <= self.classifier_dropout <= 1.0:
            raise ValueError("classifier_dropout must be between 0 and 1.")
        if not 0.0 <= self.drop_path <= 1.0:
            raise ValueError("drop_path must be between 0 and 1.")

        if self.monitor.lower() not in _MONITORS:
            raise ValueError(f"monitor must be one of {_MONITORS}.")
        if self.grad_clip_norm is not None and self.grad_clip_norm < 0:
            raise ValueError("grad_clip_norm cannot be negative.")
        if self.multi_scale and len(self.scales) == 0:
            raise ValueError("scales must be non-empty when multi_scale is enabled.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience cannot be negative.")
