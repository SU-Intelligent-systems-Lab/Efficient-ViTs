"""
This module provides:
- build_optimizer: construct an AdamW / Adam / SGD / RMSprop optimizer, with
  optional "no weight decay on norms and biases" parameter grouping.
- build_scheduler: construct a learning-rate scheduler (cosine / step /
  multistep / plateau / onecycle / constant) wrapped in a :class:`SchedulerSpec`
  that records how the trainer should step it.

Optimizers
----------
- ``adamw``  : AdamW (decoupled weight decay). The default for ViT-style models.
- ``adam``   : Adam (coupled L2 weight decay).
- ``sgd``    : SGD with Nesterov momentum.
- ``rmsprop``: RMSprop (the classic EfficientNet optimizer).

Parameter grouping
------------------
By default, weight decay is *not* applied to 1D parameters (biases and the
weights of BatchNorm/LayerNorm) nor to a model's ``no_weight_decay()`` set
(e.g. EfficientViT's relative-position bias tables). Decaying these tends to
hurt; excluding them is standard practice and makes cross-model comparison
fairer.

Schedulers and stepping cadence
-------------------------------
Different schedulers expect to be stepped at different cadences, which is a
common source of bugs. ``build_scheduler`` returns a :class:`SchedulerSpec`
that pairs the scheduler with:

- ``interval``: ``"step"`` (call ``scheduler.step()`` after every optimizer
  update) or ``"epoch"`` (call it once per epoch).
- ``needs_metric``: True for ``ReduceLROnPlateau``, which is stepped with the
  monitored validation metric.

The trainer reads these flags so each scheduler is driven correctly.

Cosine schedule
---------------
The cosine scheduler decays the learning rate along a cosine curve over the
whole run:

    lr(t) = min_lr + 0.5 * (lr_0 - min_lr) * (1 + cos(pi * t / T))

with ``t`` the optimizer step and ``T = epochs * steps_per_epoch``.

Example
-------
>>> from training.optimizer import build_optimizer, build_scheduler
>>>
>>> optimizer = build_optimizer(model, name="adamw", lr=1e-3, weight_decay=0.05)
>>> spec = build_scheduler(optimizer, "cosine", epochs=100, steps_per_epoch=500)
>>> # in the loop: if spec.interval == "step": spec.scheduler.step()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn
from torch.optim import SGD, Adam, AdamW, Optimizer, RMSprop
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    MultiStepLR,
    OneCycleLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)


__all__ = [
    "build_optimizer",
    "build_scheduler",
    "SchedulerSpec",
    "OPTIMIZERS",
    "SCHEDULERS",
]


# Names accepted by the factories (used for validation and CLI help).
OPTIMIZERS = ("adamw", "adam", "sgd", "rmsprop")
SCHEDULERS = ("cosine", "step", "multistep", "plateau", "onecycle", "constant")


def _split_param_groups(
    model: nn.Module,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """
    Split parameters into a weight-decay group and a no-weight-decay group.

    Parameters whose tensor has <= 1 dimension (biases, BatchNorm/LayerNorm
    affine weights), and any parameter named by ``model.no_weight_decay()``,
    are placed in the no-decay group. Everything else gets the configured
    weight decay.

    Returns
    -------
    list[dict]
        Two parameter-group dicts suitable for an optimizer constructor.
    """
    # Names the model explicitly wants excluded from weight decay.
    no_decay_names: set[str] = set()
    if hasattr(model, "no_weight_decay"):
        try:
            no_decay_names = set(model.no_weight_decay())  # type: ignore[attr-defined]
        except Exception:
            no_decay_names = set()

    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 1D params (biases / norm weights) or explicitly excluded names.
        if param.ndim <= 1 or name in no_decay_names or "attention_bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def build_optimizer(
    model: nn.Module,
    name: str = "adamw",
    lr: float = 1e-3,
    weight_decay: float = 0.05,
    betas: tuple[float, float] = (0.9, 0.999),
    momentum: float = 0.9,
    filter_bias_and_bn: bool = True,
) -> Optimizer:
    """
    Build an optimizer over the trainable parameters of ``model``.

    Parameters
    ----------
    model:
        Model whose parameters will be optimized.

    name:
        One of ``"adamw"``, ``"adam"``, ``"sgd"``, ``"rmsprop"`` (case insensitive).

    lr:
        Learning rate.

    weight_decay:
        Weight decay (L2 / decoupled). Applied only to the decay group when
        ``filter_bias_and_bn`` is True.

    betas:
        Beta coefficients for the Adam/AdamW running averages.

    momentum:
        Momentum for SGD and RMSprop (ignored by Adam/AdamW).

    filter_bias_and_bn:
        If True (default), exclude biases, normalization parameters, and
        ``no_weight_decay()`` parameters from weight decay via parameter groups.

    Returns
    -------
    Optimizer
        The configured optimizer.
    """
    if lr <= 0:
        raise ValueError("lr must be positive.")
    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative.")

    name = name.lower()
    if name not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {name!r}.")

    # Parameter groups: either decay/no-decay split or a single group.
    if filter_bias_and_bn:
        params: Any = _split_param_groups(model, weight_decay)
        group_weight_decay = 0.0  # per-group values already set above
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        group_weight_decay = weight_decay

    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=group_weight_decay, betas=betas)
    if name == "adam":
        return Adam(params, lr=lr, weight_decay=group_weight_decay, betas=betas)
    if name == "sgd":
        return SGD(params, lr=lr, weight_decay=group_weight_decay, momentum=momentum, nesterov=True)
    # rmsprop
    return RMSprop(params, lr=lr, weight_decay=group_weight_decay, momentum=momentum)


@dataclass
class SchedulerSpec:
    """
    A scheduler plus the metadata the trainer needs to step it correctly.

    Attributes
    ----------
    scheduler:
        The learning-rate scheduler, or None for no scheduling.

    interval:
        ``"step"`` to step after every optimizer update, ``"epoch"`` to step
        once per epoch.

    needs_metric:
        True if ``scheduler.step(metric)`` must be called with the monitored
        validation metric (ReduceLROnPlateau).
    """

    scheduler: LRScheduler | ReduceLROnPlateau | None
    interval: str = "step"
    needs_metric: bool = False


def build_scheduler(
    optimizer: Optimizer,
    name: str = "cosine",
    *,
    epochs: int,
    steps_per_epoch: int,
    lr: float | None = None,
    min_lr: float = 0.0,
    step_size: int = 30,
    milestones: tuple[int, ...] = (30, 60, 90),
    gamma: float = 0.1,
    plateau_mode: str = "max",
    plateau_factor: float = 0.1,
    plateau_patience: int = 10,
    onecycle_pct_start: float = 0.3,
    warmup_epochs: int = 0,
    warmup_start_factor: float = 0.01,
) -> SchedulerSpec:
    """
    Build a learning-rate scheduler wrapped in a :class:`SchedulerSpec`.

    Parameters
    ----------
    optimizer:
        Optimizer to schedule.

    name:
        One of :data:`SCHEDULERS`: ``"cosine"``, ``"step"``, ``"multistep"``,
        ``"plateau"``, ``"onecycle"``, ``"constant"`` (case insensitive).

    epochs:
        Total number of training epochs.

    steps_per_epoch:
        Optimizer updates per epoch (length of the train loader). Used by the
        per-iteration schedulers (cosine, onecycle).

    lr:
        Peak learning rate. Required by ``onecycle`` (its ``max_lr``); ignored
        otherwise. Defaults to the optimizer's current LR when None.

    min_lr:
        Final learning rate for the cosine schedule.

    warmup_epochs:
        Number of linear warmup epochs at the start of training. During warmup
        the LR ramps linearly from ``warmup_start_factor * lr`` up to ``lr``.
        Warmup is important for AdamW-trained transformers. It is honoured by
        the ``"cosine"`` and ``"constant"`` schedules (which then step per
        iteration); for the other schedules it is ignored. ``onecycle`` has its
        own built-in warmup via ``onecycle_pct_start``.

    warmup_start_factor:
        LR multiplier at the very first step of warmup (e.g. 0.01 -> start at
        1% of the peak LR). Must be > 0.

    step_size:
        Epoch period for ``step`` (StepLR).

    milestones:
        Epoch milestones for ``multistep`` (MultiStepLR).

    gamma:
        Multiplicative decay factor for ``step`` / ``multistep``.

    plateau_mode:
        ``"max"`` (default) or ``"min"`` for ReduceLROnPlateau, matching the
        direction of the monitored metric (accuracy/F1 -> "max", loss -> "min").

    plateau_factor, plateau_patience:
        ReduceLROnPlateau decay factor and patience (in epochs).

    onecycle_pct_start:
        Fraction of the cycle spent increasing the LR in ``onecycle``.

    Returns
    -------
    SchedulerSpec
        Scheduler plus stepping metadata.
    """
    if epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be a positive integer.")
    if min_lr < 0:
        raise ValueError("min_lr cannot be negative.")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs cannot be negative.")
    if not 0.0 < warmup_start_factor <= 1.0:
        raise ValueError("warmup_start_factor must be in (0, 1].")

    name = name.lower()
    if name not in SCHEDULERS:
        raise ValueError(f"scheduler must be one of {SCHEDULERS}, got {name!r}.")

    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    if name == "cosine":
        # Cosine decay lr_0 -> min_lr, optionally preceded by a linear warmup.
        # Both phases step every optimizer update.
        if warmup_steps > 0:
            warmup = LinearLR(
                optimizer, start_factor=warmup_start_factor, total_iters=warmup_steps
            )
            cosine = CosineAnnealingLR(
                optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=min_lr
            )
            # SequentialLR switches from warmup to cosine at the milestone step.
            scheduler = SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)
        return SchedulerSpec(scheduler, interval="step")

    if name == "onecycle":
        # The 1cycle policy: warm up then anneal; stepped every update.
        max_lr = lr if lr is not None else optimizer.param_groups[0]["lr"]
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=onecycle_pct_start,
        )
        return SchedulerSpec(scheduler, interval="step")

    if name == "step":
        # Multiply LR by gamma every step_size epochs (stepped per epoch).
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        return SchedulerSpec(scheduler, interval="epoch")

    if name == "multistep":
        scheduler = MultiStepLR(optimizer, milestones=list(milestones), gamma=gamma)
        return SchedulerSpec(scheduler, interval="epoch")

    if name == "plateau":
        # Reduce LR when the monitored metric stops improving (stepped per
        # epoch with the metric value).
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=plateau_mode,
            factor=plateau_factor,
            patience=plateau_patience,
        )
        return SchedulerSpec(scheduler, interval="epoch", needs_metric=True)

    # constant: LR held fixed for the whole run, optionally after a linear
    # warmup. ConstantLR(factor=1.0, total_iters=0) is a no-op schedule.
    if warmup_steps > 0:
        warmup = LinearLR(
            optimizer, start_factor=warmup_start_factor, total_iters=warmup_steps
        )
        held = ConstantLR(optimizer, factor=1.0, total_iters=0)
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup, held], milestones=[warmup_steps]
        )
        return SchedulerSpec(scheduler, interval="step")
    return SchedulerSpec(ConstantLR(optimizer, factor=1.0, total_iters=0), interval="epoch")
