"""
This module provides:
- MobileViTBlock: a hybrid CNN-Transformer block for image feature maps.

What it does
------------
A MobileViT block extracts local features with a small convolution, then
applies a Transformer encoder over patches to capture global relationships
between distant spatial positions, and finally fuses the result with the
original input. The block preserves spatial resolution; spatial
downsampling is done by a preceding MobileNetV2Block with stride 2.

Block structure
---------------
1. Local convolution: 3 x 3 conv extracts local features per pixel.
2. Pointwise projection: 1 x 1 conv maps C_in -> transformer_dim.
3. Patch unfolding: feature map is rearranged into Transformer tokens.
4. Transformer encoder: global attention is computed over patches.
5. Patch folding: tokens are reshaped back into a feature map.
6. Pointwise projection: 1 x 1 conv maps transformer_dim -> C_out.
7. Feature fusion: the result is concatenated with the input along the
   channel dimension and fused by a final 3 x 3 conv.

Patch tokenization (the "unfold" trick)
---------------------------------------
A standard ViT lays out tokens as one per patch and computes attention over
all of them. MobileViT instead splits each patch into its individual pixels
and groups them so the Transformer attends *across patches* for each
within-patch position separately.

Concretely, with patch size (p_h, p_w), patch_area = p_h * p_w, and a
feature map of shape (B, C, H, W) where H is divisible by p_h and W by p_w:

    n_h = H / p_h            number of patches along H
    n_w = W / p_w            number of patches along W
    N   = n_h * n_w          total number of patches

The unfold produces a tensor of shape:

    (B * patch_area, N, C)

The Transformer then runs B * patch_area independent sequences, each of
length N. Two pixels with the same within-patch offset (e.g., the top-left
pixel of every patch) all live in the same sequence and attend to each
other. After the Transformer, the fold inverts this and recovers the
original (B, C, H, W) feature map.

This keeps the attention sequence length proportional to the number of
patches rather than the number of pixels, which is what makes the
Transformer affordable on dense feature maps.

Example
-------
>>> import torch
>>> from models.mobilevit_block import MobileViTBlock
>>>
>>> x = torch.randn(2, 64, 32, 32)  # (B, C, H, W)
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

    Combines a small local convolution and a Transformer encoder over
    patches, with a residual fusion step at the end. The output has the
    same spatial size as the input.

    Parameters
    ----------
    in_channels:
        Number of input channels (C_in).

    out_channels:
        Number of output channels (C_out).

    transformer_dim:
        Channel dimension used inside the Transformer encoder. The 1x1
        projection maps C_in -> transformer_dim before tokenization.

    depth:
        Number of Transformer encoder blocks (L in the paper).

    patch_size:
        Spatial patch size, either an int p or a tuple (p_h, p_w). The
        feature map height and width are rounded up to multiples of these
        before the unfold step (and resized back at the end).

    n_heads:
        Number of attention heads in the Transformer encoder.

    mlp_ratio:
        Expansion ratio for the Transformer MLP hidden dimension.

    conv_kernel_size:
        Kernel size for the local and fusion 3x3 convs. Must be odd so the
        spatial size is preserved.

    attention_dropout:
        Dropout probability applied to attention probabilities.

    projection_dropout:
        Dropout probability applied after the attention output projection.

    mlp_dropout:
        Dropout probability used inside the Transformer MLP.

    drop_path:
        Stochastic depth probability inside Transformer encoder blocks.

    qkv_bias:
        Whether to use bias in the Q/K/V projection.

    norm_layer:
        Normalization layer used for convolutional layers. Default is BatchNorm2d.

    activation_layer:
        Activation layer used for convolutional layers. Default is SiLU.

    transformer_norm_layer:
        Normalization layer used inside the Transformer encoder. Default is LayerNorm.
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
        self.patch_area = patch_h * patch_w  # number of pixels per patch

        # "Same" padding for the odd conv_kernel_size.
        padding = conv_kernel_size // 2

        # Step 1: local k x k conv that mixes neighboring pixels but keeps
        # the channel count unchanged.
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

        # Step 2: 1 x 1 conv that lifts to the Transformer width.
        self.project_to_transformer = PointwiseConv(
            in_channels=in_channels,
            out_channels=transformer_dim,
            linear=False,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        # Step 4: Transformer encoder operating on patch tokens.
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
        # Final LayerNorm after the encoder stack.
        self.transformer_norm = transformer_norm_layer(transformer_dim)

        # Step 6: 1 x 1 conv that maps Transformer width back to C_out.
        self.project_from_transformer = PointwiseConv(
            in_channels=transformer_dim,
            out_channels=out_channels,
            linear=False,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        # Step 7: fusion. Concatenate input and Transformer-refined output
        # along channels, then mix them with a small conv. The 2*C
        # concatenation is what makes this a "fusion" rather than a
        # plain residual.
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
        Normalize ``patch_size`` to a two-value tuple (p_h, p_w).

        Examples
        --------
        patch_size = 2       -> (2, 2)
        patch_size = (2, 4)  -> (2, 4)
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
        Resize the feature map so H and W are divisible by the patch size.

        The unfold step requires H % p_h == 0 and W % p_w == 0. When the
        incoming map does not already satisfy this, we bilinearly resize
        up to the nearest larger compatible size. The forward method
        resizes back to the original size before fusion.
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

        Input shape:  (B, C, H, W)         H % p_h == 0, W % p_w == 0
        Output shape: (B * patch_area, N, C)
            where N = (H/p_h) * (W/p_w) is the number of patches.

        Each row of length N is a sequence of within-patch positions at the
        same offset across all patches, so the Transformer attends across
        patches for that offset.

        Returns the unfolded tensor and ``fold_info`` (B, C, H, W) that
        ``_fold`` needs to reverse the operation.
        """
        batch_size, channels, height, width = x.shape
        patch_h, patch_w = self.patch_size

        num_patch_h = height // patch_h
        num_patch_w = width // patch_w
        num_patches = num_patch_h * num_patch_w

        # Split H and W into (num_patches, within-patch) dimensions.
        # (B, C, H, W) -> (B, C, n_h, p_h, n_w, p_w)
        x = x.reshape(
            batch_size,
            channels,
            num_patch_h,
            patch_h,
            num_patch_w,
            patch_w,
        )

        # Move within-patch coordinates next to batch so that they become
        # independent attention streams, and put channels last.
        # (B, C, n_h, p_h, n_w, p_w) -> (B, p_h, p_w, n_h, n_w, C)
        x = x.permute(0, 3, 5, 2, 4, 1).contiguous()

        # Flatten (B, p_h, p_w) into a single stream dim and (n_h, n_w) into
        # the sequence dim.
        # -> (B * patch_area, N, C)
        x = x.reshape(batch_size * self.patch_area, num_patches, channels)

        fold_info = (batch_size, channels, height, width)
        return x, fold_info

    def _fold(self, x: Tensor, fold_info: tuple[int, int, int, int]) -> Tensor:
        """
        Inverse of ``_unfold``: rearrange Transformer tokens back into a
        feature map.

        Input shape:  (B * patch_area, N, C)
        Output shape: (B, C, H, W)
        """
        batch_size, channels, height, width = fold_info
        patch_h, patch_w = self.patch_size

        num_patch_h = height // patch_h
        num_patch_w = width // patch_w

        # Undo the reshape: split the leading stream into (B, p_h, p_w) and
        # the sequence into (n_h, n_w).
        # (B*p_h*p_w, n_h*n_w, C) -> (B, p_h, p_w, n_h, n_w, C)
        x = x.reshape(
            batch_size,
            patch_h,
            patch_w,
            num_patch_h,
            num_patch_w,
            channels,
        )

        # Reverse the permutation so channels come right after batch and
        # the patch axes are in the right order.
        # (B, p_h, p_w, n_h, n_w, C) -> (B, C, n_h, p_h, n_w, p_w)
        x = x.permute(0, 5, 3, 1, 4, 2).contiguous()

        # Merge (n_h, p_h) back into H and (n_w, p_w) back into W.
        # -> (B, C, H, W)
        x = x.reshape(batch_size, channels, height, width)

        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the MobileViT block.

        Input shape:  (B, C_in,  H, W)
        Output shape: (B, C_out, H, W)
        """
        if x.ndim != 4:
            raise ValueError(
                f"MobileViTBlock expects input shape (B, C, H, W), but got {tuple(x.shape)}."
            )

        # Keep the input around for the residual concat at the end.
        residual = x
        original_size = x.shape[-2:]

        # Steps 1-2: local conv plus 1x1 lift to transformer_dim.
        # (B, C_in, H, W) -> (B, transformer_dim, H, W)
        x = self.local_conv(x)
        x = self.project_to_transformer(x)

        # Pad up to a patch-friendly resolution if needed.
        x = self._resize_if_needed(x)

        # Steps 3-5: unfold -> Transformer -> norm -> fold.
        x, fold_info = self._unfold(x)
        x, _ = self.transformer(x)
        x = self.transformer_norm(x)
        x = self._fold(x, fold_info)

        # If we resized up for the unfold, bring the spatial size back.
        if x.shape[-2:] != original_size:
            x = F.interpolate(
                x,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )

        # Step 6: project back from Transformer width to C_out.
        x = self.project_from_transformer(x)

        # Step 7: concatenate with the original input along channels and
        # let a final 3x3 conv learn to fuse the two views.
        x = torch.cat((residual, x), dim=1)
        x = self.fusion_conv(x)

        return x
