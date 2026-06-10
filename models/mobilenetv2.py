"""
This module provides:
- PointwiseConv: 1 x 1 convolution for channel mixing, expansion, or projection.
- DepthwiseConv: spatial convolution applied independently to each channel.
- MobileNetV2Block: one MobileNetV2 inverted residual block.

Why "inverted residual"?
------------------------
A classic ResNet block has a wide-narrow-wide shape (project down, conv,
project up) with a residual on the wide tensors. MobileNetV2 inverts this:
project up (expansion), do a cheap depthwise conv on the wide tensor, then
project back down (linear projection). The residual is placed on the
narrow tensors, where memory is small.

Depthwise-separable convolution
-------------------------------
The block factors a standard convolution into two pieces:
1. DepthwiseConv (k x k, groups = channels): one spatial filter per channel.
2. PointwiseConv (1 x 1):                      mixes information across channels.

A standard k x k conv from C_in to C_out has cost ~ C_in * C_out * k^2.
The separable version costs ~ C_in * k^2 + C_in * C_out, which is much
smaller when k^2 > 1.

A MobileNetV2 block applies the depthwise convolution on an expanded
hidden tensor (so the cheap spatial part operates with more channels)
and the cheap pointwise projections expand and shrink the channel count.

Block structure
---------------
1. Pointwise expansion: 1 x 1 conv, C_in -> hidden = C_in * expand_ratio.
   Skipped when hidden == C_in (i.e. expand_ratio rounds to 1).
2. Depthwise convolution: k x k, hidden channels, optional stride 2.
3. Linear pointwise projection: 1 x 1 conv, hidden -> C_out, no activation.
4. Residual connection: added only when stride == 1 and C_in == C_out.

Example
-------
>>> import torch
>>> from models.mobilenetv2 import MobileNetV2Block
>>>
>>> x = torch.randn(2, 32, 32, 32)  # (B, C, H, W)
>>> block = MobileNetV2Block(in_channels=32, out_channels=64, stride=2)
>>>
>>> y = block(x)
>>> y.shape
torch.Size([2, 64, 16, 16])
"""

from __future__ import annotations

from torch import Tensor, nn


__all__ = [
    "PointwiseConv",
    "DepthwiseConv",
    "MobileNetV2Block",
]


class PointwiseConv(nn.Module):
    """
    Pointwise (1 x 1) convolution followed by normalization and (optional)
    activation.

    A 1 x 1 convolution does not look at neighbouring spatial positions: it
    mixes information across the channel dimension only. For each output
    pixel (b, c_out, h, w) the value is a linear combination of all input
    channels at the same spatial position:

        y[b, c_out, h, w] = sum_{c_in} W[c_out, c_in] * x[b, c_in, h, w] + bias

    Setting ``linear=True`` skips the activation; this is used as the final
    "linear bottleneck" projection in a MobileNetV2 block, where applying a
    non-linearity to a low-dimensional projection would discard information.

    Parameters
    ----------
    in_channels:
        Number of input channels (C_in).

    out_channels:
        Number of output channels (C_out).

    linear:
        If True, no activation is applied after normalization. Use this for
        the final projection in a MobileNetV2 block.

    bias:
        Whether to use bias in the convolution. Disabled by default because
        the following BatchNorm2d already learns a per-channel shift.

    norm_layer:
        Normalization layer class. Default is nn.BatchNorm2d.

    activation_layer:
        Activation layer class. Default is nn.SiLU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        linear: bool = False,
        bias: bool = False,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        activation_layer: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels must be a positive integer.")

        if out_channels <= 0:
            raise ValueError("out_channels must be a positive integer.")

        # Conv -> Norm is the standard pattern; the activation comes last
        # and is omitted when linear=True.
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            norm_layer(out_channels),
        ]

        if not linear:
            layers.append(activation_layer())

        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply pointwise convolution.

        Input shape:  (B, C_in,  H, W)
        Output shape: (B, C_out, H, W)
        """
        return self.layers(x)


class DepthwiseConv(nn.Module):
    """
    Depthwise k x k spatial convolution with normalization and activation.

    A depthwise convolution uses ``groups=channels``, which makes each
    output channel depend on a single input channel:

        y[b, c, h, w] = sum_{di, dj} W[c, di, dj] * x[b, c, h+di, w+dj]

    There is no mixing across channels. The companion PointwiseConv in
    the surrounding MobileNetV2Block does the channel mixing.

    Parameters
    ----------
    channels:
        Number of input and output channels (the layer preserves C).

    kernel_size:
        Spatial kernel size. Padding is set to kernel_size // 2 so that
        spatial size is preserved when stride == 1.

    stride:
        Spatial stride. Must be 1 (no downsampling) or 2 (halves H and W).

    bias:
        Whether to use bias in the convolution. Disabled by default because
        the following BatchNorm2d already learns a per-channel shift.

    norm_layer:
        Normalization layer class. Default is nn.BatchNorm2d.

    activation_layer:
        Activation layer class. Default is nn.SiLU.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = False,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        activation_layer: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError("channels must be a positive integer.")

        if kernel_size <= 0:
            raise ValueError("kernel_size must be a positive integer.")

        if stride not in {1, 2}:
            raise ValueError("stride must be either 1 or 2.")

        # "Same" padding for odd kernels: outputs match inputs spatially
        # when stride == 1, and are roughly halved when stride == 2.
        padding = kernel_size // 2

        self.layers = nn.Sequential(
            # groups=channels makes this a depthwise (per-channel) conv.
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=channels,
                bias=bias,
            ),
            norm_layer(channels),
            activation_layer(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply depthwise convolution.

        Input shape:  (B, C, H,     W    )
        Output shape: (B, C, H_out, W_out)

        where H_out, W_out equal H, W when stride == 1, and approximately
        H/2, W/2 when stride == 2.
        """
        return self.layers(x)


class MobileNetV2Block(nn.Module):
    """
    One MobileNetV2 inverted residual block.

    The forward path is:

        expansion (1x1) -> depthwise (kxk) -> linear projection (1x1)

    With expansion factor t = expand_ratio and hidden = round(C_in * t):

        (B, C_in, H, W)
            -> PointwiseConv:  (B, hidden, H,     W    )   (if hidden != C_in)
            -> DepthwiseConv:  (B, hidden, H/s,   W/s  )   (s = stride)
            -> PointwiseConv:  (B, C_out,  H/s,   W/s  )   (linear, no activation)

    A residual is added only when the input and output have the same shape:

        y = x + f(x)   if stride == 1 and C_in == C_out
        y =     f(x)   otherwise

    Parameters
    ----------
    in_channels:
        Number of input channels (C_in).

    out_channels:
        Number of output channels (C_out).

    stride:
        Spatial stride of the depthwise convolution. Must be 1 or 2.

    expand_ratio:
        Channel expansion ratio t. The hidden channels are
        round(in_channels * t). When t == 1 the expansion 1x1 conv is
        skipped because it would be a no-op.

    kernel_size:
        Kernel size used in the depthwise convolution (typically 3).

    norm_layer:
        Normalization layer class. Default is nn.BatchNorm2d.

    activation_layer:
        Activation layer class. Default is nn.SiLU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: float = 4.0,
        kernel_size: int = 3,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        activation_layer: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels must be a positive integer.")

        if out_channels <= 0:
            raise ValueError("out_channels must be a positive integer.")

        if stride not in {1, 2}:
            raise ValueError("stride must be either 1 or 2.")

        if expand_ratio <= 0:
            raise ValueError("expand_ratio must be positive.")

        # hidden = C_in * t, rounded to the nearest integer.
        hidden_channels = int(round(in_channels * expand_ratio))

        # The residual is valid only when input and output shapes match,
        # so spatial size must be preserved (stride 1) and channel count
        # must be unchanged.
        self.use_residual = stride == 1 and in_channels == out_channels

        layers: list[nn.Module] = []

        # Step 1 (optional): pointwise expansion C_in -> hidden.
        # Skip when expansion would be a no-op (hidden == C_in).
        if hidden_channels != in_channels:
            layers.append(
                PointwiseConv(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    linear=False,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # Step 2: depthwise spatial filtering on the wide hidden tensor.
        layers.append(
            DepthwiseConv(
                channels=hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # Step 3: linear pointwise projection hidden -> C_out.
        # No activation here: this is the "linear bottleneck" projection.
        layers.append(
            PointwiseConv(
                in_channels=hidden_channels,
                out_channels=out_channels,
                linear=True,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply one MobileNetV2 block.

        Input shape:  (B, C_in,  H,     W    )
        Output shape: (B, C_out, H_out, W_out)
        """
        # Main branch: expansion, depthwise spatial filtering, projection.
        out = self.layers(x)

        # Add the residual only when shapes match exactly.
        if self.use_residual:
            return x + out

        return out
