"""
Training configuration for the compact MobileViT library.

This module provides:
- TrainConfig: simple dataclass for one training run

Example
-------
>>> from mobilevit.training.config import TrainConfig
>>>
>>> config = TrainConfig(
...     epochs=100,
...     batch_size=128,
...     lr=1e-3,
...     num_classes=100,
... )
"""

from __future__ import annotations

from dataclasses import dataclass


__all__ = ["TrainConfig"]


@dataclass(frozen=True)
class TrainConfig:
    """
    Configuration for one training run.

    Parameters
    ----------
    epochs:
        Number of training epochs.

    batch_size:
        Training batch size.

    num_classes:
        Number of output classes.

    lr:
        Learning rate.

    weight_decay:
        Weight decay used by the optimizer.

    optimizer:
        Optimizer name. Supported values are "adamw" and "sgd".

    min_lr:
        Minimum learning rate for cosine scheduling.

    label_smoothing:
        Label smoothing value passed to torch.nn.CrossEntropyLoss.

    output_dir:
        Directory used for saving checkpoints.

    device:
        Device string, for example, "cuda" or "cpu". If None, Trainer chooses automatically.
    """

    epochs: int = 100
    batch_size: int = 128
    num_classes: int = 1000

    lr: float = 1e-3
    weight_decay: float = 0.05
    optimizer: str = "adamw"
    min_lr: float = 0.0

    label_smoothing: float = 0.0
    classifier_dropout: float = 0.0
    drop_path: float = 0.0

    output_dir: str = "checkpoints"
    device: str | None = None
    num_workers: int = 4
    verbose: bool = True
    log_interval: int = 50

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive.")

        if self.lr <= 0:
            raise ValueError("lr must be positive.")

        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")

        if self.optimizer not in {"adamw", "sgd"}:
            raise ValueError("optimizer must be 'adamw' or 'sgd'.")

        if self.min_lr < 0:
            raise ValueError("min_lr cannot be negative.")

        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1).")

        if not 0.0 <= self.classifier_dropout <= 1.0:
            raise ValueError("classifier_dropout must be between 0 and 1.")

        if not 0.0 <= self.drop_path <= 1.0:
            raise ValueError("drop_path must be between 0 and 1.")

        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")

        if self.log_interval < 0:
            raise ValueError("log_interval cannot be negative.")
