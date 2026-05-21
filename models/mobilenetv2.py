"""
This module provides:
- PointwiseConv: 1 x 1 convolution for channel mixing, expansion, or projection
- DepthwiseConv: spatial convolution applied independently to each channel
- MobileNetV2Block: one complete MobileNetV2 inverted residual block

A MobileNetV2 block follows this structure:
1. Pointwise expansion: 1 x 1 convolution expands the number of channels.
2. Depthwise convolution: spatial convolution is applied independently to each channel.
3. Linear pointwise projection: 1 x 1 convolution projects features to output channels.
4. Residual connection: used only when stride = 1 and input/output channels match.

Example
-------
>>> import torch
>>> from models.mobilenetv2 import MobileNetV2Block
>>>
>>> x = torch.randn(2, 32, 32, 32)  # (batch, channels, height, width)
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
    Pointwise 1 x 1 convolution.

    This layer mixes information across channels. It can be used either as an
    expansion layer with activation, or as a linear projection without activation.

    Parameters
    ----------
    in_channels:
        Number of input channels.

    out_channels:
        Number of output channels.

    linear:
        If True, no activation is used after normalization.
        This is used for the final projection in MobileNetV2.

    bias:
        Whether to use bias in the convolution. Usually False when followed by BatchNorm2d.

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

        Input shape: (B, C, H, W)
        Output shape: (B, C_out, H, W)
        """
        return self.layers(x)


class DepthwiseConv(nn.Module):
    """
    Depthwise spatial convolution.

    This layer applies one spatial convolution per channel. It does not mix
    information across channels.

    Parameters
    ----------
    channels:
        Number of input and output channels.

    kernel_size:
        Spatial kernel size of the depthwise convolution.

    stride:
        Spatial stride. Usually 1 or 2.

    bias:
        Whether to use bias in the convolution. Usually False when followed by BatchNorm2d.

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

        padding = kernel_size // 2

        self.layers = nn.Sequential(
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

        Input shape: (B, C, H, W)
        Output shape: (B, C, H_out, W_out)
        """
        return self.layers(x)


class MobileNetV2Block(nn.Module):
    """
    One MobileNetV2 inverted residual block.

    Structure:
    1. Optional pointwise expansion
    2. Depthwise spatial convolution
    3. Linear pointwise projection
    4. Optional residual connection

    Parameters
    ----------
    in_channels:
        Number of input channels.

    out_channels:
        Number of output channels.

    stride:
        Spatial stride of the depthwise convolution. Usually 1 or 2.

    expand_ratio:
        Channel expansion ratio. hidden_channels = in_channels * expand_ratio.

    kernel_size:
        Kernel size used in the depthwise convolution.

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

        hidden_channels = int(round(in_channels * expand_ratio))
        self.use_residual = stride == 1 and in_channels == out_channels

        layers: list[nn.Module] = []

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

        layers.append(
            DepthwiseConv(
                channels=hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

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

        Input shape: (B, C, H, W)
        Output shape: (B, C_out, H_out, W_out)
        """
        # Main branch: expansion, depthwise spatial filtering, and projection.
        out = self.layers(x)

        # Residual is valid only when spatial size and channel size are unchanged.
        if self.use_residual:
            return x + out

        return out
