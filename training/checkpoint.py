"""
This module provides:
- save_model: save only model weights
- load_model: load only model weights

Example
-------
>>> from mobilevit.training.checkpoint import save_model, load_model
>>>
>>> save_model("mobilevit.pt", model)
>>> load_model("mobilevit.pt", model)
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
    Save only the model weights.

    Parameters
    ----------
    path:
        File path where the model weights will be saved.

    model:
        Model to save.
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
    Load model weights.

    Parameters
    ----------
    path:
        File path of the saved model weights.

    model:
        Model whose weights will be loaded.

    device:
        Device used for loading. If None, PyTorch chooses the default.

    strict:
        Whether to strictly enforce that checkpoint keys match model keys.

    Returns
    -------
    nn.Module
        Model with loaded weights.
    """
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict, strict=strict)
    return model
