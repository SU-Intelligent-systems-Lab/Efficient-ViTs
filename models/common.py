"""
Small shared helpers used across every model family in this library.

This module deliberately contains only *lightweight, dependency-free*
utilities so that every other ``models/*`` file can import from it without
creating import cycles. It provides three groups of helpers:

1. Validation
   ``ensure_positive_int`` / ``ensure_positive_float`` / ``ensure_probability``
   raise ``ValueError`` with a consistent, readable message. Centralising the
   checks keeps the validation wording identical across MobileViT,
   EfficientViT, EfficientFormer, and the torchvision wrappers.

2. Channel arithmetic
   ``make_divisible`` rounds a channel count to a multiple of ``divisor``.
   This is the standard MobileNet/EfficientNet trick that keeps tensor widths
   hardware-friendly (most kernels prefer channel counts that are multiples
   of 8) while never dropping more than ~10% of the requested width.

3. Parameters and initialization
   ``count_parameters`` reports model size; ``init_module_weights`` applies the
   Vision-Transformer-style initialization scheme shared by the hand-written
   models in this repository.

Initialization scheme
---------------------
``init_module_weights`` initialises one ``nn.Module`` according to its type:

- ``nn.Conv2d``        -> Kaiming-normal (``fan_out``, ``relu``), bias 0.
- ``BatchNorm`` / ``LayerNorm`` / ``GroupNorm`` -> weight 1, bias 0 (identity).
- ``nn.Linear``        -> truncated normal (``std=0.02``), bias 0.

Conv initialisation can be disabled with ``init_conv=False``. That matches
EfficientFormer, whose reference implementation only re-initialises Linear and
LayerNorm layers and leaves convolutions at PyTorch's default. Pass the helper
to ``model.apply(...)`` to walk an entire module tree.

Example
-------
>>> import torch
>>> from torch import nn
>>> from models.common import count_parameters, init_module_weights, make_divisible
>>>
>>> make_divisible(58, divisor=8)
56
>>> net = nn.Sequential(nn.Conv2d(3, 16, 3), nn.BatchNorm2d(16), nn.Linear(16, 10))
>>> net.apply(init_module_weights)            # in-place, returns the module
Sequential(...)
>>> count_parameters(net) > 0
True
"""

from __future__ import annotations

from torch import nn


__all__ = [
    "ensure_positive_int",
    "ensure_positive_float",
    "ensure_probability",
    "make_divisible",
    "count_parameters",
    "count_parameters_millions",
    "init_module_weights",
]


def ensure_positive_int(value: int, name: str) -> int:
    """
    Validate that ``value`` is a positive integer.

    Parameters
    ----------
    value:
        The value to check.

    name:
        Human-readable parameter name used in the error message.

    Returns
    -------
    int
        The validated value, returned unchanged so the call can be inlined,
        for example ``self.dim = ensure_positive_int(dim, "dim")``.

    Raises
    ------
    ValueError
        If ``value`` is not a strictly positive integer. ``bool`` is rejected
        explicitly because ``True``/``False`` are ints in Python and would
        otherwise slip through.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value


def ensure_positive_float(value: float, name: str) -> float:
    """
    Validate that ``value`` is a strictly positive real number.

    Accepts ``int`` or ``float`` (an int such as ``4`` is a valid float here).
    Raises ``ValueError`` otherwise.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number, got {value!r}.")
    return float(value)


def ensure_probability(value: float, name: str) -> float:
    """
    Validate that ``value`` lies in the closed interval ``[0, 1]``.

    Used for dropout rates, stochastic-depth rates, and any other probability.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number in [0, 1], got {value!r}.")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value!r}.")
    return float(value)


def make_divisible(value: float, divisor: int = 8, min_value: int | None = None) -> int:
    """
    Round ``value`` to the nearest multiple of ``divisor``.

    This is the channel-rounding rule introduced by MobileNetV2 and reused by
    EfficientNet. Hardware convolution kernels run fastest when the channel
    count is a multiple of 8, so widths produced by a width multiplier are
    snapped to such a multiple. The rounding never decreases the value by more
    than 10%, which prevents an aggressive round-down from silently shrinking a
    layer:

        new = max(min_value, round(value / divisor) * divisor)
        if new < 0.9 * value:          # rounded down too much
            new += divisor

    Parameters
    ----------
    value:
        The (possibly fractional) channel count to round.

    divisor:
        The multiple to round to. Default 8.

    min_value:
        Lower bound on the result. Defaults to ``divisor`` when ``None``.

    Returns
    -------
    int
        The rounded channel count, always a positive multiple of ``divisor``.
    """
    ensure_positive_int(divisor, "divisor")
    if min_value is None:
        min_value = divisor

    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Do not let the rounding drop more than 10% of the requested width.
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of scalar parameters in ``model``.

    Parameters
    ----------
    model:
        The module to inspect.

    trainable_only:
        If True (default), count only parameters with ``requires_grad=True``.
        If False, count every parameter regardless of gradient status.

    Returns
    -------
    int
        Total number of scalar entries summed over the selected parameter
        tensors (``sum(p.numel() for p in ...)``).
    """
    parameters = (
        (p for p in model.parameters() if p.requires_grad)
        if trainable_only
        else model.parameters()
    )
    return sum(p.numel() for p in parameters)


def count_parameters_millions(model: nn.Module, trainable_only: bool = True) -> float:
    """
    Return ``count_parameters`` divided by 1e6, for readable model-size reports.
    """
    return count_parameters(model, trainable_only=trainable_only) / 1e6


def init_module_weights(module: nn.Module, init_conv: bool = True) -> None:
    """
    Initialise a single layer using the shared Vision-Transformer scheme.

    Intended to be passed to ``model.apply(init_module_weights)``, which calls
    it once per submodule. Only the leaf layer types listed below are touched;
    container modules are ignored.

    - ``nn.Conv2d`` (only when ``init_conv`` is True):
        Kaiming-normal with ``mode="fan_out"`` and ``nonlinearity="relu"``,
        a good default for ReLU/SiLU/GELU networks. Bias (if any) set to 0.
    - ``nn.BatchNorm2d`` / ``nn.BatchNorm1d`` / ``nn.LayerNorm`` / ``nn.GroupNorm``:
        weight set to 1 and bias to 0 so the layer starts as the identity.
    - ``nn.Linear``:
        truncated normal with ``std=0.02`` (the standard ViT choice), bias 0.

    Parameters
    ----------
    module:
        The submodule to initialise in place.

    init_conv:
        Whether to (re)initialise ``nn.Conv2d`` weights. Set to False to leave
        convolutions at PyTorch's default initialisation (matches the
        EfficientFormer reference, which only re-inits Linear and LayerNorm).

    Notes
    -----
    ``apply`` cannot pass extra arguments, so to disable conv init within an
    ``apply`` call use a small lambda::

        model.apply(lambda m: init_module_weights(m, init_conv=False))
    """
    if init_conv and isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(
        module,
        (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm),
    ):
        # Affine norm layers may have weight/bias set to None when affine=False.
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
