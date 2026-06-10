"""
This module provides:
- save_model: save model weights (``state_dict``) to disk.
- load_model: load model weights from disk into an existing model.

Checkpointing policy
--------------------
This library saves **only the best model's weights** -- a single ``best.pt``
file containing nothing but the model ``state_dict``. Optimizer state, scheduler
state, epoch counters, and metric history are deliberately *not* saved, because
the goal is later inference / evaluation / latency comparison, for which the
weights alone suffice. The file stays small and loads into a freshly built model
of the same architecture.

(There is intentionally no ``last.pt`` and no full training checkpoint; the
trainer overwrites ``best.pt`` whenever the monitored validation metric
improves.)

Example
-------
>>> from training.checkpoint import save_model, load_model
>>>
>>> save_model("best.pt", model)
>>> load_model("best.pt", model)
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


__all__ = [
    "save_model",
    "load_model",
]


def save_model(
    path: str | Path,
    model: nn.Module,
) -> None:
    """
    Save only the model weights (``state_dict``) to ``path``.

    Creates the parent directory if it does not already exist.

    Parameters
    ----------
    path:
        File path where the model weights will be saved (e.g. ``best.pt``).

    model:
        Model whose ``state_dict`` will be serialized.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), path)


def load_model(
    path: str | Path,
    model: nn.Module,
    device: str | torch.device | None = None,
    strict: bool = True,
) -> nn.Module:
    """
    Load model weights from ``path`` into ``model`` in place.

    Parameters
    ----------
    path:
        File path of the saved model weights.

    model:
        Model whose weights will be replaced. Must have the same
        architecture as the model used when the checkpoint was saved.

    device:
        Device used as ``map_location`` for ``torch.load``. If None, the
        tensors are loaded onto the device they were saved on.

    strict:
        Whether to strictly enforce that the checkpoint's parameter keys
        match the model's parameter keys. Set False to load partially
        compatible checkpoints.

    Returns
    -------
    nn.Module
        The same ``model`` instance, with its weights replaced.
    """
    # weights_only=True is safe here: checkpoints are pure tensor state_dicts.
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=strict)
    return model
