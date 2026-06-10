"""
This module provides:
- EfficientFormer: the full EfficientFormer classification model.
- efficientformer_l1, efficientformer_l3, efficientformer_l7: paper-exact
  variants.
- efficientformer_l3_mini, efficientformer_l7_mini: depth-reduced variants for
  tractable training on small datasets.

EfficientFormer (Li et al., NeurIPS 2022) is a four-stage hierarchical network
that follows a "dimension-consistent" design: it runs almost entirely in the
hardware-friendly 4D convolutional form (``MB4D`` blocks) and switches to 3D
token attention (``MB3D`` blocks) only for the last few blocks of the final
stage, where the spatial resolution is small and attention is cheap.

Forward path
------------
```
Input (B, 3, img_size, img_size)
  └─ ConvStem (two 3x3 stride-2 convs)              -> (B, C0, img_size/4, .)
  └─ stage 1: depth0 x MB4D                          @ img_size/4
  └─ downsample (3x3 stride 2)                       -> img_size/8
  └─ stage 2: depth1 x MB4D                          @ img_size/8
  └─ downsample                                      -> img_size/16
  └─ stage 3: depth2 x MB4D                          @ img_size/16
  └─ downsample                                      -> img_size/32
  └─ stage 4: (depth3 - num_mb3d) x MB4D, then       @ img_size/32
              num_mb3d x MB3D   (4D -> 3D reshape at the boundary)
  └─ LayerNorm over channels, mean over tokens       -> (B, C3)
  └─ Linear head                                     -> (B, num_classes)
```

Only the *last* ``num_mb3d`` blocks of stage 4 are MB3D; everything before them
is MB4D. The single 4D->3D reshape happens exactly once, at the first MB3D
block.

Fixed input size
----------------
MB3D attention uses a learned ``(num_heads, N, N)`` bias with ``N`` fixed at
construction (``N = (img_size / 32)^2``). EfficientFormer is therefore a
fixed-resolution model: ``forward`` validates the input size and the registry
excludes it from multi-scale sampling. ``img_size`` must be a multiple of 32 so
every inter-stage downsample halves an even resolution cleanly.

Example
-------
>>> import torch
>>> from models.efficientformer import efficientformer_l1
>>>
>>> model = efficientformer_l1(num_classes=1000, img_size=224)
>>> x = torch.randn(2, 3, 224, 224)
>>> model(x).shape
torch.Size([2, 1000])
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .common import ensure_positive_int, init_module_weights
from .configs import (
    EfficientFormerConfig,
    efficientformer_l1_config,
    efficientformer_l3_config,
    efficientformer_l3_mini_config,
    efficientformer_l7_config,
    efficientformer_l7_mini_config,
)
from .efficientformer_block import MB3D, MB4D, ConvStem, StageDownsample


__all__ = [
    "EfficientFormer",
    "efficientformer_l1",
    "efficientformer_l3",
    "efficientformer_l7",
    "efficientformer_l3_mini",
    "efficientformer_l7_mini",
]


class EfficientFormer(nn.Module):
    """
    EfficientFormer classification model (four hierarchical stages).

    Parameters
    ----------
    config:
        Architecture configuration (see
        :class:`models.configs.EfficientFormerConfig`): per-stage widths and
        depths, the number of trailing MB3D blocks, and attention hyper-params.

    num_classes:
        Number of output classes. Pass ``0`` for a headless model.

    img_size:
        Input image side length. Must be a multiple of 32. Determines the
        stage-4 token-grid resolution and therefore the MB3D attention bias
        size. Default 224.

    in_channels:
        Number of input image channels. Default 3.
    """

    def __init__(
        self,
        config: EfficientFormerConfig,
        num_classes: int = 1000,
        img_size: int = 224,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        ensure_positive_int(in_channels, "in_channels")
        if num_classes < 0:
            raise ValueError("num_classes must be non-negative (0 builds a headless model).")
        if img_size % 32 != 0:
            raise ValueError(
                f"img_size must be a multiple of 32 (got {img_size}); the four "
                "downsampling steps each halve the resolution."
            )

        self.config = config
        self.num_classes = num_classes
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_features = config.embed_dims[-1]

        embed_dims = config.embed_dims
        depths = config.depths
        num_mb3d = config.num_mb3d

        # MB3D blocks only live at the tail of stage 4 (paper constraint).
        if not 0 <= num_mb3d <= depths[3]:
            raise ValueError(
                f"num_mb3d ({num_mb3d}) must be between 0 and the stage-4 depth "
                f"({depths[3]})."
            )

        # Linearly increasing stochastic-depth schedule across all blocks, the
        # DeiT/Swin convention: drop_path_i = drop_path_rate * i / (total - 1).
        total_blocks = sum(depths)
        dpr = torch.linspace(0, config.drop_path_rate, total_blocks).tolist()

        # Conv stem: img_size -> img_size / 4, channels -> C0.
        self.stem = ConvStem(in_channels, embed_dims[0])
        current_resolution = img_size // 4

        # Build the four stages plus the three inter-stage downsamplers.
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        block_idx = 0

        for stage_idx in range(4):
            stage_blocks = nn.ModuleList()
            stage_depth = depths[stage_idx]

            # In stage 4, the last ``num_mb3d`` blocks are MB3D; everywhere else
            # all blocks are MB4D.
            mb3d_start = stage_depth - num_mb3d if stage_idx == 3 else stage_depth

            for b in range(stage_depth):
                drop_path = dpr[block_idx]
                if b >= mb3d_start:
                    stage_blocks.append(
                        MB3D(
                            dim=embed_dims[stage_idx],
                            num_heads=config.num_heads,
                            qk_dim=config.qk_dim,
                            mlp_ratio=config.mlp_ratio,
                            resolution=current_resolution,
                            drop_path=drop_path,
                        )
                    )
                else:
                    stage_blocks.append(
                        MB4D(
                            dim=embed_dims[stage_idx],
                            mlp_ratio=config.mlp_ratio,
                            drop_path=drop_path,
                        )
                    )
                block_idx += 1

            self.stages.append(stage_blocks)

            # Downsample before the next stage (not after the last stage).
            if stage_idx < 3:
                self.downsamples.append(
                    StageDownsample(embed_dims[stage_idx], embed_dims[stage_idx + 1])
                )
                current_resolution = current_resolution // 2

        # Final LayerNorm (the tail of stage 4 is already in 3D token form) and
        # the linear classification head.
        self.norm = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        # Initialise Linear and LayerNorm layers (trunc-normal / identity);
        # leave the convolutions at PyTorch's default, matching the reference.
        # The LayerScale gammas and the attention bias (raw Parameters) are not
        # touched and keep their constructor init.
        self.apply(lambda module: init_module_weights(module, init_conv=False))

    def reset_classifier(self, num_classes: int) -> None:
        """
        Replace the classification head with a fresh ``nn.Linear``.

        Parameters
        ----------
        num_classes:
            New number of output classes. Pass ``<= 0`` to drop the head.
        """
        self.num_classes = num_classes
        if num_classes <= 0:
            self.head = nn.Identity()
            return
        self.head = nn.Linear(self.num_features, num_classes)
        self.head.apply(lambda module: init_module_weights(module, init_conv=False))

    def forward_features(self, x: Tensor) -> Tensor:
        """
        Run the stem and four stages and return the pooled feature vector.

        The pooling here is a mean over the stage-4 tokens after a final
        LayerNorm, matching the reference implementation, so the returned tensor
        is already the input to the classifier head.

        Input shape:  (B, in_channels, img_size, img_size)
        Output shape: (B, embed_dims[-1])
        """
        x = self.stem(x)  # (B, C0, img_size/4, img_size/4)

        for stage_idx, stage in enumerate(self.stages):
            # Every stage begins in 4D (B, C, H, W) form.
            in_4d = True
            for block in stage:
                if isinstance(block, MB3D) and in_4d:
                    # One-time 4D -> 3D reshape at the MB4D -> MB3D boundary.
                    # (B, C, H, W) -> (B, H*W, C)
                    batch, channels, height, width = x.shape
                    x = x.flatten(2).transpose(1, 2)
                    in_4d = False
                x = block(x)

            # Downsamplers operate in 4D. With the standard configs only stage 4
            # ever leaves 4D form, so this reshape-back is a safety net.
            if stage_idx < 3:
                if not in_4d:
                    batch, num_tokens, channels = x.shape
                    side = int(num_tokens**0.5)
                    x = x.transpose(1, 2).reshape(batch, channels, side, side)
                x = self.downsamples[stage_idx](x)

        # If the last stage stayed 4D (num_mb3d == 0), flatten to tokens now.
        if x.ndim == 4:
            x = x.flatten(2).transpose(1, 2)

        x = self.norm(x)
        # Global average pool over the token dimension -> (B, C).
        return x.mean(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Run the full EfficientFormer model.

        Input shape:  (B, in_channels, img_size, img_size)
        Output shape: (B, num_classes)
        """
        if x.ndim != 4:
            raise ValueError(
                f"EfficientFormer expects input shape (B, C, H, W), got {tuple(x.shape)}."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"EfficientFormer was built for in_channels={self.in_channels}, "
                f"got {x.shape[1]}."
            )
        if x.shape[2] != self.img_size or x.shape[3] != self.img_size:
            raise ValueError(
                "EfficientFormer is a fixed-resolution model: expected input of "
                f"size {(self.img_size, self.img_size)}, got {tuple(x.shape[2:])}. "
                "Rebuild the model with the desired img_size."
            )

        x = self.forward_features(x)
        return self.head(x)


def efficientformer_l1(
    num_classes: int = 1000, img_size: int = 224, **kwargs: Any
) -> EfficientFormer:
    """Build EfficientFormer-L1 (paper-exact; ~12M params)."""
    return EfficientFormer(efficientformer_l1_config(), num_classes, img_size, **kwargs)


def efficientformer_l3(
    num_classes: int = 1000, img_size: int = 224, **kwargs: Any
) -> EfficientFormer:
    """Build EfficientFormer-L3 (paper-exact; ~31M params)."""
    return EfficientFormer(efficientformer_l3_config(), num_classes, img_size, **kwargs)


def efficientformer_l7(
    num_classes: int = 1000, img_size: int = 224, **kwargs: Any
) -> EfficientFormer:
    """Build EfficientFormer-L7 (paper-exact; ~82M params)."""
    return EfficientFormer(efficientformer_l7_config(), num_classes, img_size, **kwargs)


def efficientformer_l3_mini(
    num_classes: int = 1000, img_size: int = 224, **kwargs: Any
) -> EfficientFormer:
    """Build EfficientFormer-L3-mini (depth-reduced L3 for small datasets)."""
    return EfficientFormer(efficientformer_l3_mini_config(), num_classes, img_size, **kwargs)


def efficientformer_l7_mini(
    num_classes: int = 1000, img_size: int = 224, **kwargs: Any
) -> EfficientFormer:
    """Build EfficientFormer-L7-mini (depth-reduced L7 for small datasets)."""
    return EfficientFormer(efficientformer_l7_mini_config(), num_classes, img_size, **kwargs)
