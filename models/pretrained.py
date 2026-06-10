"""
Local pretrained-weight registry for the hand-built model families.

The torchvision baselines ship ImageNet-1k weights, but MobileViT, EfficientViT,
and EfficientFormer do not — so this library bundles the weights that were
*trained here* (on ImageNet-100) under ``weights/<model_name>.pt`` and makes them
loadable through ``create_model(..., pretrained=True)``.

Each file is a plain ``state_dict`` (written by ``training.checkpoint.save_model``)
named exactly after its registry model name, e.g. ``weights/efficientvit_m0.pt``.
This module discovers those files and reads the head shape to recover the number
of classes the checkpoint was trained with, so the registry can build a matching
model, load the weights, and (optionally) swap in a fresh head for a different
class count.

What lives here
---------------
- ``WEIGHTS_DIR``          : the ``weights/`` directory at the package root.
- ``PRETRAINED_IMAGE_SIZE``: the resolution the bundled weights were trained at.
- ``get_pretrained_path``  : path to a model's local weights, or None.
- ``has_pretrained``       : whether local weights exist for a model.
- ``list_pretrained``      : the model names that have local weights.
- ``infer_num_classes``    : read the class count from a state_dict's head.

Example
-------
>>> from models.pretrained import list_pretrained, has_pretrained
>>> has_pretrained("efficientvit_m0")        # doctest: +SKIP
True
>>> "mobilevit_xs" in list_pretrained()      # doctest: +SKIP
True
"""

from __future__ import annotations

from pathlib import Path

import torch


__all__ = [
    "WEIGHTS_DIR",
    "PRETRAINED_IMAGE_SIZE",
    "get_pretrained_path",
    "has_pretrained",
    "list_pretrained",
    "infer_num_classes",
]


# The bundled weights live in <package root>/weights/<model_name>.pt.
# models/pretrained.py -> models/ -> <package root>.
WEIGHTS_DIR: Path = Path(__file__).resolve().parent.parent / "weights"

# The bundled checkpoints were all trained on ImageNet-100 at 224x224. This is
# the resolution at which fixed-resolution models (EfficientViT/EfficientFormer)
# must be rebuilt for their attention-bias tables to match the saved weights.
PRETRAINED_IMAGE_SIZE: int = 224

# State-dict keys that hold the final classifier weight, in priority order, one
# per family:
#   - MobileViT:        ``classifier`` (nn.Linear)
#   - EfficientViT:     ``head`` is a BNLinear -> ``head.linear``
#   - EfficientFormer:  ``head`` (nn.Linear)
_HEAD_WEIGHT_KEYS: tuple[str, ...] = (
    "classifier.weight",
    "head.linear.weight",
    "head.weight",
)


def get_pretrained_path(name: str) -> Path | None:
    """
    Return the path to ``name``'s local weights file, or None if absent.

    Parameters
    ----------
    name:
        Registry model name (e.g. ``"mobilevit_xs"``).

    Returns
    -------
    Path | None
        ``weights/<name>.pt`` if it exists, otherwise None.
    """
    path = WEIGHTS_DIR / f"{name}.pt"
    return path if path.is_file() else None


def has_pretrained(name: str) -> bool:
    """Return True if local pretrained weights exist for ``name``."""
    return get_pretrained_path(name) is not None


def list_pretrained() -> list[str]:
    """Return the sorted model names that have local weights in ``weights/``."""
    if not WEIGHTS_DIR.is_dir():
        return []
    return sorted(p.stem for p in WEIGHTS_DIR.glob("*.pt"))


def infer_num_classes(state_dict: dict[str, torch.Tensor]) -> int:
    """
    Recover the number of classes from a checkpoint's classifier weight.

    The classifier weight has shape ``(num_classes, num_features)``; we read its
    first dimension. The candidate key names cover all three hand-built families
    (see ``_HEAD_WEIGHT_KEYS``).

    Parameters
    ----------
    state_dict:
        A model ``state_dict`` (mapping parameter names to tensors).

    Returns
    -------
    int
        The number of output classes the checkpoint was trained with.

    Raises
    ------
    KeyError
        If no recognised classifier-weight key is present.
    """
    for key in _HEAD_WEIGHT_KEYS:
        if key in state_dict:
            return int(state_dict[key].shape[0])
    raise KeyError(
        "Could not find a classifier weight in the checkpoint (looked for "
        f"{_HEAD_WEIGHT_KEYS}). Keys present include: "
        f"{list(state_dict)[:8]} ..."
    )
