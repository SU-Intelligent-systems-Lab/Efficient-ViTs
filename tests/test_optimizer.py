"""Tests for optimizer/scheduler factories, including LR warmup."""

import warnings

import pytest
import torch
from torch.optim import SGD, Adam, AdamW, RMSprop

from models import create_model
from training.optimizer import OPTIMIZERS, SCHEDULERS, build_optimizer, build_scheduler


def _model():
    return create_model("mobilevit_xxs", num_classes=5)


@pytest.mark.parametrize("name,cls", [
    ("adamw", AdamW), ("adam", Adam), ("sgd", SGD), ("rmsprop", RMSprop),
])
def test_build_optimizer_types_and_param_groups(name, cls):
    opt = build_optimizer(_model(), name=name, lr=1e-3, weight_decay=0.05)
    assert isinstance(opt, cls)
    # Two groups: one with weight decay, one without (norms/biases/bias tables).
    assert len(opt.param_groups) == 2
    decays = sorted(g["weight_decay"] for g in opt.param_groups)
    assert decays[0] == 0.0 and decays[1] == 0.05


def test_build_optimizer_rejects_unknown():
    with pytest.raises(ValueError):
        build_optimizer(_model(), name="nope")
    assert set(OPTIMIZERS) == {"adamw", "adam", "sgd", "rmsprop"}


@pytest.mark.parametrize("name,interval,needs_metric", [
    ("cosine", "step", False),
    ("onecycle", "step", False),
    ("step", "epoch", False),
    ("multistep", "epoch", False),
    ("plateau", "epoch", True),
    ("constant", "epoch", False),
])
def test_build_scheduler_specs(name, interval, needs_metric):
    opt = build_optimizer(_model(), "adamw", lr=1e-3)
    spec = build_scheduler(opt, name, epochs=2, steps_per_epoch=4, lr=1e-3)
    assert spec.interval == interval
    assert spec.needs_metric is needs_metric
    assert set(SCHEDULERS) == {"cosine", "step", "multistep", "plateau", "onecycle", "constant"}


def test_cosine_warmup_ramps_then_decays():
    opt = build_optimizer(_model(), "sgd", lr=1.0, weight_decay=0.0)
    steps_per_epoch = 10
    warmup_epochs = 2  # 20 warmup steps
    spec = build_scheduler(
        opt, "cosine", epochs=10, steps_per_epoch=steps_per_epoch,
        lr=1.0, min_lr=0.0, warmup_epochs=warmup_epochs, warmup_start_factor=0.01,
    )
    # Warmup makes cosine step-interval and chained via SequentialLR.
    assert spec.interval == "step"

    lrs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore "step before optimizer.step" notice
        for _ in range(100):
            lrs.append(opt.param_groups[0]["lr"])
            spec.scheduler.step()

    warmup_steps = warmup_epochs * steps_per_epoch
    # LR starts near warmup_start_factor and ramps up across the warmup window.
    assert lrs[0] < lrs[warmup_steps // 2] < lrs[warmup_steps - 1]
    # Peak is near the base LR; never exceeds it.
    assert max(lrs) <= 1.0 + 1e-6
    # After warmup the cosine schedule decays.
    assert lrs[-1] < lrs[warmup_steps]


def test_warmup_validation():
    opt = build_optimizer(_model(), "adamw", lr=1e-3)
    with pytest.raises(ValueError):
        build_scheduler(opt, "cosine", epochs=2, steps_per_epoch=4, warmup_epochs=-1)
    with pytest.raises(ValueError):
        build_scheduler(opt, "cosine", epochs=2, steps_per_epoch=4, warmup_start_factor=0.0)
