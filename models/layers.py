"""
Reusable convolutional building blocks shared by several model families.

This module collects the small "conv + normalization (+ activation)" layers
that show up again and again across EfficientViT and the torchvision-style
models, plus a Squeeze-and-Excitation channel-attention block. Keeping them in
one place means the BatchNorm-folding logic and the channel-attention maths are
written and documented exactly once.

What lives here
---------------
- ``ConvBN``       : Conv2d (bias-free) followed by BatchNorm2d, with a
                     ``fuse()`` method that folds the two into a single Conv2d
                     for faster inference.
- ``ConvNormAct``  : ``ConvBN`` plus an activation -- the generic
                     "conv-norm-act" unit.
- ``BNLinear``     : BatchNorm1d followed by Linear, with a ``fuse()`` method.
                     Used as EfficientViT's classification head.
- ``SqueezeExcite``: Squeeze-and-Excitation channel attention (a dependency-free
                     re-implementation of the ``timm`` block EfficientViT uses).

Why fold BatchNorm into the convolution?
----------------------------------------
At inference time a ``Conv2d -> BatchNorm2d`` pair computes, per output channel
``o``:

    y = bn_gamma_o * (conv(x)_o - mu_o) / sqrt(var_o + eps) + bn_beta_o

Because BatchNorm is an affine function of the convolution output, it can be
absorbed into the convolution's own weight and bias:

    w'_o = conv_w_o * gamma_o / sqrt(var_o + eps)
    b'_o = beta_o - mu_o * gamma_o / sqrt(var_o + eps)

The fused ``Conv2d`` with ``(w', b')`` produces identical outputs with one
fewer layer. This is purely an inference-speed optimisation and changes no
results; training always uses the un-fused ``ConvBN``.

Example
-------
>>> import torch
>>> from models.layers import ConvNormAct, SqueezeExcite
>>>
>>> x = torch.randn(2, 16, 32, 32)
>>> block = ConvNormAct(16, 32, kernel_size=3, stride=2, padding=1)
>>> block(x).shape
torch.Size([2, 32, 16, 16])
>>> se = SqueezeExcite(32, rd_ratio=0.25)
>>> se(block(x)).shape
torch.Size([2, 32, 16, 16])
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .common import ensure_positive_int, make_divisible


__all__ = [
    "ConvBN",
    "ConvNormAct",
    "BNLinear",
    "SqueezeExcite",
]


class ConvBN(nn.Sequential):
    """
    Bias-free 2D convolution followed by BatchNorm2d.

    The convolution has ``bias=False`` because the following BatchNorm already
    contributes a learnable per-channel shift, so a separate conv bias would be
    redundant. ``bn_weight_init`` lets callers initialise the BatchNorm scale to
    a value other than 1: EfficientViT initialises the *final* BN of a residual
    branch to 0 so each block starts as an identity mapping (``x + 0``), which
    stabilises early training of deep residual stacks.

    Parameters
    ----------
    in_channels:
        Number of input channels.

    out_channels:
        Number of output channels.

    kernel_size:
        Spatial kernel size (square). Default 1 (pointwise).

    stride:
        Convolution stride. Default 1.

    padding:
        Zero-padding added on both spatial sides. Default 0.

    dilation:
        Convolution dilation. Default 1.

    groups:
        Number of blocked connections. ``groups == in_channels == out_channels``
        gives a depthwise convolution. Default 1.

    bn_weight_init:
        Initial value for the BatchNorm weight (gamma). Default 1.0. Use 0.0
        for the last layer of a residual branch.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bn_weight_init: float = 1.0,
    ) -> None:
        super().__init__()
        ensure_positive_int(in_channels, "in_channels")
        ensure_positive_int(out_channels, "out_channels")

        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(out_channels))

        # Initialise the BN affine parameters explicitly (gamma to
        # bn_weight_init, beta to 0). This is what makes the "init last BN to
        # zero" residual trick possible.
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self) -> nn.Conv2d:
        """
        Fold the BatchNorm into the convolution and return a single Conv2d.

        See the module docstring for the algebra. The returned layer is a plain
        ``nn.Conv2d`` (with bias) that produces numerically identical outputs in
        ``eval`` mode. Intended for inference only.
        """
        conv, bn = self.conv, self.bn

        # Per-output-channel BN scale: gamma / sqrt(var + eps). Shape (C_out,).
        scale = bn.weight / (bn.running_var + bn.eps) ** 0.5

        # Fold scale into the conv weight: broadcast over (C_in/groups, kH, kW).
        fused_weight = conv.weight * scale[:, None, None, None]
        # Fold the BN shift into a new conv bias.
        fused_bias = bn.bias - bn.running_mean * scale

        fused = nn.Conv2d(
            in_channels=fused_weight.size(1) * conv.groups,
            out_channels=fused_weight.size(0),
            kernel_size=fused_weight.shape[2:],
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
        )
        fused.weight.data.copy_(fused_weight)
        fused.bias.data.copy_(fused_bias)
        return fused


class ConvNormAct(nn.Sequential):
    """
    Convolution -> normalization -> activation, the generic CNN unit.

    A convenience wrapper around :class:`ConvBN` that appends an activation.
    Padding defaults to ``kernel_size // 2`` ("same" padding for odd kernels),
    so with ``stride=1`` the spatial size is preserved and with ``stride=2`` it
    is roughly halved.

    Parameters
    ----------
    in_channels, out_channels, kernel_size, stride, groups:
        Forwarded to the convolution. See :class:`ConvBN`.

    padding:
        Zero-padding. When ``None`` (default) it is set to ``kernel_size // 2``.

    norm_layer:
        Normalization layer class applied after the convolution.
        Default ``nn.BatchNorm2d``.

    activation_layer:
        Activation layer class applied last. Default ``nn.ReLU``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        activation_layer: type[nn.Module] = nn.ReLU,
    ) -> None:
        ensure_positive_int(in_channels, "in_channels")
        ensure_positive_int(out_channels, "out_channels")
        if padding is None:
            padding = kernel_size // 2

        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_channels),
            activation_layer(),
        )


class BNLinear(nn.Sequential):
    """
    BatchNorm1d followed by a Linear layer (EfficientViT's classifier head).

    Normalising the pooled feature vector with BatchNorm before the final
    Linear layer is the LeViT/EfficientViT choice; it stabilises the scale of
    the features entering the classifier. Like :class:`ConvBN`, the BatchNorm
    can be folded into the Linear layer at inference time via :meth:`fuse`.

    Parameters
    ----------
    in_features:
        Size of the pooled feature vector.

    out_features:
        Number of output classes.

    bias:
        Whether the Linear layer has a bias term. Default True.

    std:
        Standard deviation for the truncated-normal initialisation of the
        Linear weight. Default 0.02 (the standard ViT value).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        std: float = 0.02,
    ) -> None:
        super().__init__()
        ensure_positive_int(in_features, "in_features")
        ensure_positive_int(out_features, "out_features")

        self.add_module("bn", nn.BatchNorm1d(in_features))
        self.add_module("linear", nn.Linear(in_features, out_features, bias=bias))

        nn.init.trunc_normal_(self.linear.weight, std=std)
        if bias:
            nn.init.constant_(self.linear.bias, 0)

    @torch.no_grad()
    def fuse(self) -> nn.Linear:
        """
        Fold the BatchNorm1d into the Linear layer and return a single Linear.

        For a ``BatchNorm1d -> Linear`` pair the BN is an affine map on the
        input features, so it can be absorbed into the Linear weight and bias.
        Returns a plain ``nn.Linear`` giving identical ``eval``-mode outputs.
        """
        bn, linear = self.bn, self.linear

        # BN as an affine map x -> scale * x + shift on the input features.
        scale = bn.weight / (bn.running_var + bn.eps) ** 0.5
        shift = bn.bias - bn.running_mean * scale

        # Compose Linear(BN(x)) = (W * scale) x + (W @ shift + b).
        fused_weight = linear.weight * scale[None, :]
        if linear.bias is None:
            fused_bias = linear.weight @ shift
        else:
            fused_bias = (linear.weight @ shift) + linear.bias

        fused = nn.Linear(fused_weight.size(1), fused_weight.size(0))
        fused.weight.data.copy_(fused_weight)
        fused.bias.data.copy_(fused_bias)
        return fused


class SqueezeExcite(nn.Module):
    """
    Squeeze-and-Excitation channel attention (Hu et al., 2018).

    A lightweight, dependency-free re-implementation of the SE block used by
    EfficientViT's ``PatchMerging`` (originally imported from ``timm``). The
    block learns a per-channel gate in ``[0, 1]`` and rescales the input
    channels by it:

        1. Squeeze:  global average pool over space -> (B, C, 1, 1).
        2. Reduce:   1x1 conv  C -> C_rd  (C_rd = C * rd_ratio, rounded), ReLU.
        3. Expand:   1x1 conv  C_rd -> C.
        4. Gate:     sigmoid  -> per-channel weights in [0, 1].
        5. Scale:    multiply the input by the gate (broadcast over H, W).

    The reduction bottleneck (``rd_ratio < 1``) keeps the block cheap relative
    to the convolutions it augments.

    Parameters
    ----------
    channels:
        Number of input/output channels ``C``.

    rd_ratio:
        Reduction ratio for the squeeze bottleneck. The reduced width is
        ``make_divisible(channels * rd_ratio, 8)``. Default 0.25.

    act_layer:
        Activation between the reduce and expand convolutions. Default ReLU.

    gate_layer:
        Gating activation producing the channel weights. Default Sigmoid.
    """

    def __init__(
        self,
        channels: int,
        rd_ratio: float = 0.25,
        act_layer: type[nn.Module] = nn.ReLU,
        gate_layer: type[nn.Module] = nn.Sigmoid,
    ) -> None:
        super().__init__()
        ensure_positive_int(channels, "channels")
        if not 0.0 < rd_ratio <= 1.0:
            raise ValueError(f"rd_ratio must be in (0, 1], got {rd_ratio!r}.")

        # Reduced (bottleneck) channel count, snapped to a multiple of 8.
        rd_channels = make_divisible(channels * rd_ratio, divisor=8)

        self.conv_reduce = nn.Conv2d(channels, rd_channels, kernel_size=1, bias=True)
        self.act = act_layer()
        self.conv_expand = nn.Conv2d(rd_channels, channels, kernel_size=1, bias=True)
        self.gate = gate_layer()

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply channel attention.

        Input shape:  (B, C, H, W)
        Output shape: (B, C, H, W)  -- same shape, channels rescaled by the gate.
        """
        # Squeeze: average each channel over all spatial positions.
        # (B, C, H, W) -> (B, C, 1, 1)
        x_se = x.mean((2, 3), keepdim=True)
        # Reduce -> activate -> expand back to C channels.
        x_se = self.conv_reduce(x_se)
        x_se = self.act(x_se)
        x_se = self.conv_expand(x_se)
        # Gate to [0, 1] and rescale the input channels (broadcast over H, W).
        return x * self.gate(x_se)
