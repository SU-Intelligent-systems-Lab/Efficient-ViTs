"""
Shared stochastic-depth utilities.

This module provides the regularisation primitives that every model family in
this library reuses, so the exact same "drop an entire residual branch"
behaviour is shared between MobileViT's Transformer blocks, EfficientViT's
residual wrappers, and EfficientFormer's MetaBlocks:

- ``drop_path``  : the functional stochastic-depth operation.
- ``DropPath``   : an ``nn.Module`` wrapper around ``drop_path``.
- ``Residual``   : wraps a sub-module ``f`` and returns ``x + drop_path(f(x))``.

Stochastic depth
----------------
Stochastic depth (Huang et al., 2016) drops *whole residual branches* for a
random subset of the samples in a batch, rather than dropping individual
activations the way ``nn.Dropout`` does. For a residual block ``y = x + f(x)``
it turns the update into:

    keep_prob   = 1 - drop_prob
    mask_i      ~ Bernoulli(keep_prob)          # one draw per sample i
    y_i         = x_i + (mask_i / keep_prob) * f(x_i)

The division by ``keep_prob`` is the "inverted dropout" trick: it makes
``E[y_i] == x_i + f(x_i)`` in expectation, so downstream layers see the same
average magnitude during training and at evaluation time. At eval, or when
``drop_prob == 0``, the branch is passed through unchanged.

The mask has shape ``(B, 1, 1, ..., 1)`` so the single per-sample Bernoulli
draw broadcasts across every non-batch dimension. This makes ``drop_path``
rank-agnostic: it works for ``(B, S, E)`` Transformer tokens and for
``(B, C, H, W)`` convolutional feature maps alike.

Example
-------
>>> import torch
>>> from models.stochastic import DropPath, Residual
>>> from torch import nn
>>>
>>> x = torch.randn(4, 32, 8, 8)
>>> block = Residual(nn.Conv2d(32, 32, 3, padding=1), drop_prob=0.1)
>>> block(x).shape
torch.Size([4, 32, 8, 8])
>>>
>>> drop = DropPath(drop_prob=0.2)
>>> drop.eval()(x) is x      # eval is a no-op
True
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .common import ensure_probability


__all__ = [
    "drop_path",
    "DropPath",
    "Residual",
]


def drop_path(x: Tensor, drop_prob: float, training: bool) -> Tensor:
    """
    Apply stochastic depth to ``x`` (functional form).

    Parameters
    ----------
    x:
        Input tensor. The first dimension must be the batch dimension; any
        number of trailing dimensions is allowed.

    drop_prob:
        Probability of dropping the branch for a given sample, in ``[0, 1]``.

    training:
        Whether the surrounding module is in training mode. When False (or when
        ``drop_prob == 0``) the input is returned unchanged.

    Returns
    -------
    Tensor
        Same shape as ``x``. Surviving samples are scaled by ``1 / keep_prob``.
    """
    # No-op at eval time or when there is nothing to drop. We compare to 0.0
    # exactly because that is the overwhelmingly common configured value.
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1.0 - drop_prob

    # Mask shape (B, 1, 1, ..., 1): one Bernoulli draw per sample, broadcast
    # over every feature dimension so an entire branch is kept or dropped.
    mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)

    # bernoulli_(keep_prob) -> 1 with prob keep_prob else 0; div_ by keep_prob
    # rescales the surviving samples (inverted-dropout), keeping E[output] = x.
    mask = x.new_empty(mask_shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        mask.div_(keep_prob)
    return x * mask


class DropPath(nn.Module):
    """
    Stochastic-depth regularisation as an ``nn.Module``.

    A thin wrapper over :func:`drop_path` so it can be placed inside an
    ``nn.Sequential`` or assigned as a submodule. Typically applied to the
    *output of a residual branch* before it is added back to the skip
    connection (see :class:`Residual`).

    Parameters
    ----------
    drop_prob:
        Probability of dropping the branch for a given sample. Must be in
        ``[0, 1]``. A value of 0 makes the module an identity.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = ensure_probability(drop_prob, "drop_prob")

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply stochastic depth.

        Input shape:  ``(B, ...)`` -- the leading dimension must be the batch.
        Output shape: identical to the input.
        """
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


class Residual(nn.Module):
    """
    Residual wrapper with optional stochastic depth.

    Computes ``y = x + drop_path(f(x))`` where ``f`` is the wrapped module.
    This is the residual primitive used throughout EfficientViT: every
    depthwise conv, FFN, and attention mixer in an EfficientViT block is
    wrapped in a ``Residual`` so the block is a clean stack of residual
    sub-layers.

    Because the addition ``x + f(x)`` requires ``f`` to preserve the shape of
    ``x``, the wrapped module must map ``(B, C, H, W) -> (B, C, H, W)`` (or
    whatever the input shape is) without changing it.

    Parameters
    ----------
    fn:
        The residual branch. Must return a tensor with the same shape as its
        input.

    drop_prob:
        Stochastic-depth probability applied to the branch output during
        training. Must be in ``[0, 1]``.
    """

    def __init__(self, fn: nn.Module, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.fn = fn
        self.drop_prob = ensure_probability(drop_prob, "drop_prob")

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the residual branch.

        Input shape:  ``(B, ...)``
        Output shape: identical to the input (``f`` must be shape-preserving).
        """
        return x + drop_path(self.fn(x), self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"
