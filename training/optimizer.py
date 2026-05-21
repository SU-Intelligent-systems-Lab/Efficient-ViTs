"""
This module provides:
- build_optimizer: creates AdamW or SGD
- build_scheduler: creates a cosine learning-rate scheduler

Example
-------
>>> from mobilevit.training.optimizer import build_optimizer, build_scheduler
>>>
>>> optimizer = build_optimizer(model, name="adamw", lr=1e-3, weight_decay=0.05)
>>> scheduler = build_scheduler(optimizer, epochs=100, steps_per_epoch=500)
"""

from __future__ import annotations

from torch import nn
from torch.optim import AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR


__all__ = [
    "build_optimizer",
    "build_scheduler",
]


def build_optimizer(
    model: nn.Module,
    name: str = "adamw",
    lr: float = 1e-3,
    weight_decay: float = 0.05,
    betas: tuple[float, float] = (0.9, 0.999),
    momentum: float = 0.9,
) -> Optimizer:
    """
    Build an optimizer.

    Parameters
    ----------
    model:
        Model whose parameters will be optimized.

    name:
        Optimizer name. Supported values are "adamw" and "sgd".

    lr:
        Learning rate.

    weight_decay:
        Weight decay used by the optimizer.

    betas:
        AdamW beta values.

    momentum:
        SGD momentum value.
    """
    if lr <= 0:
        raise ValueError("lr must be positive.")

    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative.")

    parameters = model.parameters()
    name = name.lower()

    if name == "adamw":
        return AdamW(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
        )

    if name == "sgd":
        return SGD(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=True,
        )

    raise ValueError("optimizer name must be either 'adamw' or 'sgd'.")


def build_scheduler(
    optimizer: Optimizer,
    epochs: int,
    steps_per_epoch: int,
    min_lr: float = 0.0,
) -> CosineAnnealingLR:
    """
    Build a cosine learning-rate scheduler.

    The scheduler is stepped once per optimizer update.

    Parameters
    ----------
    optimizer:
        Optimizer whose learning rate will be scheduled.

    epochs:
        Total number of training epochs.

    steps_per_epoch:
        Number of optimizer updates per epoch.

    min_lr:
        Minimum learning rate at the end of cosine decay.
    """
    if epochs <= 0:
        raise ValueError("epochs must be a positive integer.")

    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be a positive integer.")

    if min_lr < 0:
        raise ValueError("min_lr cannot be negative.")

    total_steps = epochs * steps_per_epoch

    return CosineAnnealingLR(
        optimizer=optimizer,
        T_max=total_steps,
        eta_min=min_lr,
    )
