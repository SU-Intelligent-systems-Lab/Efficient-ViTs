"""
This module provides:
- MobileNetV2BlockConfig: configuration for one MobileNetV2 inverted residual block.
- MobileViTBlockConfig: configuration for one MobileViT hybrid CNN-Transformer block.
- MobileViTConfig: full architecture configuration, described as a flat
  sequence of backbone blocks.
- mobilevit_s_config, mobilevit_xs_config, mobilevit_xxs_config: factory
  functions returning the three standard MobileViT variants from the paper.

A MobileViT model is described as:
1. Stem: a 3 x 3 stride-2 convolution that maps the input image to feature maps.
2. Backbone: a flat sequence of MobileNetV2 blocks and MobileViT blocks,
   applied in the order given by MobileViTConfig.blocks.
3. Final pointwise convolution: a 1 x 1 convolution applied before the
   classification head.

Variants
--------
Three official variants are provided:
- MobileViT-S:   ~5.6M parameters
- MobileViT-XS:  ~2.3M parameters
- MobileViT-XXS: ~1.3M parameters

Example
-------
>>> from models.configs import mobilevit_s_config
>>> config = mobilevit_s_config()
>>> config.final_conv_channels
640
>>> len(config.blocks)
10
"""

from __future__ import annotations

from dataclasses import dataclass


__all__ = [
    "MobileNetV2BlockConfig",
    "MobileViTBlockConfig",
    "BackboneBlockConfig",
    "MobileViTConfig",
    "mobilevit_s_config",
    "mobilevit_xs_config",
    "mobilevit_xxs_config",
]


@dataclass(frozen=True)
class MobileNetV2BlockConfig:
    """
    Configuration for one MobileNetV2 inverted residual block.

    Parameters
    ----------
    out_channels:
        Number of output channels for the block.

    stride:
        Spatial stride of the depthwise convolution. Must be 1 or 2.
        A stride of 2 halves the feature-map spatial resolution.
    """

    out_channels: int
    stride: int

    def __post_init__(self) -> None:
        if self.out_channels <= 0:
            raise ValueError("out_channels must be a positive integer.")

        if self.stride not in {1, 2}:
            raise ValueError("stride must be either 1 or 2.")


@dataclass(frozen=True)
class MobileViTBlockConfig:
    """
    Configuration for one MobileViT hybrid CNN-Transformer block.

    A MobileViT block preserves the spatial resolution of the feature map.
    Down-sampling, when required between blocks, is performed by a preceding
    MobileNetV2 block.

    Parameters
    ----------
    out_channels:
        Number of output channels for the block.

    transformer_dim:
        Channel dimension used inside the Transformer encoder of the block.

    transformer_depth:
        Number of Transformer encoder blocks inside the MobileViT block.
        This is L in the paper.
    """

    out_channels: int
    transformer_dim: int
    transformer_depth: int

    def __post_init__(self) -> None:
        if self.out_channels <= 0:
            raise ValueError("out_channels must be a positive integer.")

        if self.transformer_dim <= 0:
            raise ValueError("transformer_dim must be a positive integer.")

        if self.transformer_depth <= 0:
            raise ValueError("transformer_depth must be a positive integer.")


BackboneBlockConfig = MobileNetV2BlockConfig | MobileViTBlockConfig


@dataclass(frozen=True)
class MobileViTConfig:
    """
    Full configuration for a MobileViT model.

    The backbone is described as a flat sequence of block configurations.
    Each entry is exactly one block, in the order in which it is applied to
    the feature map.

    Parameters
    ----------
    stem_channels:
        Number of output channels of the initial 3 x 3 stride-2 convolution.

    blocks:
        Flat sequence of MobileNetV2BlockConfig and MobileViTBlockConfig.
        Each entry describes one backbone block.

    final_conv_channels:
        Number of output channels of the 1 x 1 convolution applied after the
        backbone and before the classifier.

    expand_ratio:
        Expansion ratio used by every MobileNetV2 block in the backbone.

    patch_size:
        Spatial patch size used by every MobileViT block.

    n_heads:
        Number of attention heads used inside MobileViT blocks.

    mlp_ratio:
        Expansion ratio for the Transformer MLP inside MobileViT blocks.
    """

    stem_channels: int
    blocks: tuple[BackboneBlockConfig, ...]
    final_conv_channels: int
    expand_ratio: float = 4.0
    patch_size: int | tuple[int, int] = 2
    n_heads: int = 4
    mlp_ratio: float = 2.0

    def __post_init__(self) -> None:
        if self.stem_channels <= 0:
            raise ValueError("stem_channels must be a positive integer.")

        if self.final_conv_channels <= 0:
            raise ValueError("final_conv_channels must be a positive integer.")

        if len(self.blocks) == 0:
            raise ValueError("blocks must contain at least one entry.")

        if self.expand_ratio <= 0:
            raise ValueError("expand_ratio must be positive.")

        if self.n_heads <= 0:
            raise ValueError("n_heads must be a positive integer.")

        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive.")

        if isinstance(self.patch_size, int):
            if self.patch_size <= 0:
                raise ValueError("patch_size must be a positive integer.")
        elif isinstance(self.patch_size, tuple):
            if len(self.patch_size) != 2:
                raise ValueError("patch_size must contain exactly two integers.")
            if any(not isinstance(size, int) for size in self.patch_size):
                raise ValueError("patch_size values must be integers.")
            if any(size <= 0 for size in self.patch_size):
                raise ValueError("patch_size values must be positive integers.")
        else:
            raise ValueError("patch_size must be an integer or a tuple of two integers.")

        for block in self.blocks:
            if isinstance(block, MobileViTBlockConfig):
                if block.transformer_dim % self.n_heads != 0:
                    raise ValueError(
                        "MobileViTBlockConfig.transformer_dim must be divisible by n_heads."
                    )


def mobilevit_s_config() -> MobileViTConfig:
    """
    Return the standard MobileViT-S configuration from the paper.

    Has approximately 5.6M parameters.
    """
    return MobileViTConfig(
        stem_channels=16,
        blocks=(
            MobileNetV2BlockConfig(out_channels=32, stride=1),
            MobileNetV2BlockConfig(out_channels=64, stride=2),
            MobileNetV2BlockConfig(out_channels=64, stride=1),
            MobileNetV2BlockConfig(out_channels=64, stride=1),
            MobileNetV2BlockConfig(out_channels=96, stride=2),
            MobileViTBlockConfig(
                out_channels=96,
                transformer_dim=144,
                transformer_depth=2,
            ),
            MobileNetV2BlockConfig(out_channels=128, stride=2),
            MobileViTBlockConfig(
                out_channels=128,
                transformer_dim=192,
                transformer_depth=4,
            ),
            MobileNetV2BlockConfig(out_channels=160, stride=2),
            MobileViTBlockConfig(
                out_channels=160,
                transformer_dim=240,
                transformer_depth=3,
            ),
        ),
        final_conv_channels=640,
        expand_ratio=4.0,
    )


def mobilevit_xs_config() -> MobileViTConfig:
    """
    Return the standard MobileViT-XS configuration from the paper.

    Has approximately 2.3M parameters.
    """
    return MobileViTConfig(
        stem_channels=16,
        blocks=(
            MobileNetV2BlockConfig(out_channels=32, stride=1),
            MobileNetV2BlockConfig(out_channels=48, stride=2),
            MobileNetV2BlockConfig(out_channels=48, stride=1),
            MobileNetV2BlockConfig(out_channels=48, stride=1),
            MobileNetV2BlockConfig(out_channels=64, stride=2),
            MobileViTBlockConfig(
                out_channels=64,
                transformer_dim=96,
                transformer_depth=2,
            ),
            MobileNetV2BlockConfig(out_channels=80, stride=2),
            MobileViTBlockConfig(
                out_channels=80,
                transformer_dim=120,
                transformer_depth=4,
            ),
            MobileNetV2BlockConfig(out_channels=96, stride=2),
            MobileViTBlockConfig(
                out_channels=96,
                transformer_dim=144,
                transformer_depth=3,
            ),
        ),
        final_conv_channels=384,
        expand_ratio=4.0,
    )


def mobilevit_xxs_config() -> MobileViTConfig:
    """
    Return the standard MobileViT-XXS configuration from the paper.

    Has approximately 1.3M parameters.
    """
    return MobileViTConfig(
        stem_channels=16,
        blocks=(
            MobileNetV2BlockConfig(out_channels=16, stride=1),
            MobileNetV2BlockConfig(out_channels=24, stride=2),
            MobileNetV2BlockConfig(out_channels=24, stride=1),
            MobileNetV2BlockConfig(out_channels=24, stride=1),
            MobileNetV2BlockConfig(out_channels=48, stride=2),
            MobileViTBlockConfig(
                out_channels=48,
                transformer_dim=64,
                transformer_depth=2,
            ),
            MobileNetV2BlockConfig(out_channels=64, stride=2),
            MobileViTBlockConfig(
                out_channels=64,
                transformer_dim=80,
                transformer_depth=4,
            ),
            MobileNetV2BlockConfig(out_channels=80, stride=2),
            MobileViTBlockConfig(
                out_channels=80,
                transformer_dim=96,
                transformer_depth=3,
            ),
        ),
        final_conv_channels=320,
        expand_ratio=2.0,
    )
