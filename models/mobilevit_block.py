"""
This module provides:
- MobileViTBlock: a hybrid CNN-Transformer block for image feature maps

A MobileViT block follows this structure:
1. Local convolution: spatial convolution extracts local features.
2. Pointwise projection: 1 x 1 convolution maps channels to transformer_dim.
3. Patch unfolding: the feature map is rearranged into transformer tokens.
4. Transformer encoder: global relationships are modeled across patches.
5. Patch folding: tokens are reshaped back into a feature map with transformer_dim channels.
6. Pointwise projection: 1 x 1 convolution maps transformer_dim to output channels.
7. Feature fusion: transformed features are concatenated with the input and fused.

The block preserves spatial resolution. Downsampling should be handled before this
block using MobileNetV2Block with stride = 2.

Example
-------
>>> import torch
>>> from models.mobilevit_block import MobileViTBlock
>>>
>>> x = torch.randn(2, 64, 32, 32)  # (batch, channels, height, width)
>>> block = MobileViTBlock(
...     in_channels=64,
...     out_channels=64,
...     transformer_dim=96,
...     depth=2,
...     patch_size=2,
...     n_heads=4,
... )
>>>
>>> y = block(x)
>>> y.shape
torch.Size([2, 64, 32, 32])
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .mobilenetv2 import PointwiseConv
from .transformer import TransformerEncoder


__all__ = [
    "MobileViTBlock",
]


class MobileViTBlock(nn.Module):
    """
    Hybrid CNN-Transformer block used in MobileViT.

    Parameters
    ----------
    in_channels:
        Number of input channels.

    out_channels:
        Number of output channels.

    transformer_dim:
        Channel dimension used inside the Transformer encoder after pointwise projection.

    depth:
        Number of Transformer encoder blocks.

    patch_size:
        Spatial patch size. It Can be an integer or a tuple of two integers.

    n_heads:
        Number of attention heads in the Transformer encoder.

    mlp_ratio:
        Expansion ratio for the Transformer MLP.

    conv_kernel_size:
        Kernel size used by the local and fusion convolutions.
        It must be odd, so spatial size is preserved.

    attention_dropout:
        Dropout probability applied to attention probabilities.

    projection_dropout:
        Dropout probability applied after attention output projection.

    mlp_dropout:
        Dropout probability used inside the Transformer MLP.

    drop_path:
        Stochastic depth probability used inside Transformer encoder blocks.

    qkv_bias:
        Whether to use bias in the query/key/value projection.

    norm_layer:
        Normalization layer used for convolutional layers. Default is nn.BatchNorm2d.

    activation_layer:
        Activation layer used for convolutional layers. Default is nn.SiLU.

    transformer_norm_layer:
        Normalization layer used inside the Transformer encoder. Default is nn.LayerNorm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        transformer_dim: int,
        depth: int,
        patch_size: int | tuple[int, int] = 2,
        n_heads: int = 4,
        mlp_ratio: float = 2.0,
        conv_kernel_size: int = 3,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        drop_path: float = 0.0,
        qkv_bias: bool = True,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        activation_layer: type[nn.Module] = nn.SiLU,
        transformer_norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels must be a positive integer.")

        if out_channels <= 0:
            raise ValueError("out_channels must be a positive integer.")

        if transformer_dim <= 0:
            raise ValueError("transformer_dim must be a positive integer.")

        if conv_kernel_size <= 0:
            raise ValueError("conv_kernel_size must be a positive integer.")

        if conv_kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size must be odd to preserve spatial size.")

        patch_h, patch_w = self._normalize_patch_size(patch_size)
        self.patch_size = (patch_h, patch_w)
        self.patch_area = patch_h * patch_w

        padding = conv_kernel_size // 2

        self.local_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=conv_kernel_size,
                stride=1,
                padding=padding,
                bias=False,
            ),
            norm_layer(in_channels),
            activation_layer(),
        )

        self.project_to_transformer = PointwiseConv(
            in_channels=in_channels,
            out_channels=transformer_dim,
            linear=False,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        self.transformer = TransformerEncoder(
            embed_dim=transformer_dim,
            depth=depth,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
            mlp_dropout=mlp_dropout,
            drop_path=drop_path,
            qkv_bias=qkv_bias,
            activation_layer=nn.GELU,
            norm_layer=transformer_norm_layer,
        )
        self.transformer_norm = transformer_norm_layer(transformer_dim)

        self.project_from_transformer = PointwiseConv(
            in_channels=transformer_dim,
            out_channels=out_channels,
            linear=False,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels + out_channels,
                out_channels=out_channels,
                kernel_size=conv_kernel_size,
                stride=1,
                padding=padding,
                bias=False,
            ),
            norm_layer(out_channels),
            activation_layer(),
        )

    @staticmethod
    def _normalize_patch_size(patch_size: int | tuple[int, int]) -> tuple[int, int]:
        """
        Convert patch_size to a two-value tuple.

        Examples
        --------
        patch_size = 2 becomes (2, 2)
        patch_size = (2, 4) stays (2, 4)
        """
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        if not isinstance(patch_size, tuple) or len(patch_size) != 2:
            raise ValueError("patch_size must be an integer or a tuple of two integers.")

        patch_h, patch_w = patch_size

        if not isinstance(patch_h, int) or not isinstance(patch_w, int):
            raise ValueError("patch dimensions must be integers.")

        if patch_h <= 0 or patch_w <= 0:
            raise ValueError("patch dimensions must be positive integers.")

        return patch_h, patch_w

    def _resize_if_needed(self, x: Tensor) -> Tensor:
        """
        Resize feature maps so height and width are divisible by patch_size.

        The unfolding step requires H to be divisible by patch_h and W to be
        divisible by patch_w. If this is not true, the feature map is resized
        to the nearest larger compatible size.

        The forward method later resizes the folded feature map back to the
        original input size before feature fusion.
        """
        patch_h, patch_w = self.patch_size
        height, width = x.shape[-2:]

        new_height = math.ceil(height / patch_h) * patch_h
        new_width = math.ceil(width / patch_w) * patch_w

        if new_height == height and new_width == width:
            return x

        return F.interpolate(
            x,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )

    def _unfold(self, x: Tensor) -> tuple[Tensor, tuple[int, int, int, int]]:
        """
        Rearrange a feature map into Transformer tokens.

        Input shape:
        (B, C, H, W)

        Output shape:
        (B * patch_area, num_patches, C)

        Interpretation:
        - num_patches is the number of spatial patches in the feature map.
        - patch_area is patch_h * patch_w.
        - Each position inside a patch becomes a separate batch stream.
        - The Transformer attends across patches for each within-patch position.

        This is the compact MobileViT-style tokenization used to keep the
        Transformer lightweight.
        """
        batch_size, channels, height, width = x.shape
        patch_h, patch_w = self.patch_size

        num_patch_h = height // patch_h
        num_patch_w = width // patch_w
        num_patches = num_patch_h * num_patch_w

        x = x.reshape(
            batch_size,
            channels,
            num_patch_h,
            patch_h,
            num_patch_w,
            patch_w,
        )
        x = x.permute(0, 3, 5, 2, 4, 1).contiguous()
        x = x.reshape(batch_size * self.patch_area, num_patches, channels)

        fold_info = (batch_size, channels, height, width)
        return x, fold_info

    def _fold(self, x: Tensor, fold_info: tuple[int, int, int, int]) -> Tensor:
        """
        Rearrange Transformer tokens back into a feature map.

        Input shape:
        (B * patch_area, num_patches, C)

        Output shape:
        (B, C, H, W)

        fold_info stores the original batch size, channel count, height, and
        width of the resized feature map before unfolding.
        """
        batch_size, channels, height, width = fold_info
        patch_h, patch_w = self.patch_size

        num_patch_h = height // patch_h
        num_patch_w = width // patch_w

        x = x.reshape(
            batch_size,
            patch_h,
            patch_w,
            num_patch_h,
            num_patch_w,
            channels,
        )
        x = x.permute(0, 5, 3, 1, 4, 2).contiguous()
        x = x.reshape(batch_size, channels, height, width)

        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the MobileViT block.

        Input shape: (B, C_in, H, W)
        Output shape: (B, C_out, H, W)
        """
        if x.ndim != 4:
            raise ValueError(
                f"MobileViTBlock expects input shape (B, C, H, W), but got {tuple(x.shape)}."
            )

        residual = x
        original_size = x.shape[-2:]

        # Local CNN representation.
        x = self.local_conv(x)
        x = self.project_to_transformer(x)

        # Patch-compatible size for unfold/fold.
        x = self._resize_if_needed(x)

        # Global Transformer representation.
        x, fold_info = self._unfold(x)
        x, _ = self.transformer(x)
        x = self.transformer_norm(x)
        x = self._fold(x, fold_info)

        # Restore the original spatial size before fusion with the residual input.
        if x.shape[-2:] != original_size:
            x = F.interpolate(
                x,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )

        # Project Transformer features back to CNN channels and fuse with input.
        x = self.project_from_transformer(x)
        x = torch.cat((residual, x), dim=1)
        x = self.fusion_conv(x)

        return x
