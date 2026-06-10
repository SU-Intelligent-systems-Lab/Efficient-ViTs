"""
This module provides:
- EfficientViT: the full EfficientViT classification model.
- efficientvit_m0 .. efficientvit_m5: factory functions for the six standard
  variants (Liu et al., CVPR 2023).

EfficientViT is a three-stage hierarchical vision transformer that stays in the
spatial ``(B, C, H, W)`` layout throughout. Its design goal is *memory-bound*
efficiency: cheap depthwise convolutions and grouped/cascaded attention reduce
the memory traffic that dominates latency on real hardware.

Forward path
------------
```
Input (B, 3, img_size, img_size)
  └─ PatchEmbed: four 3x3 stride-2 convs            -> (B, C0, R0, R0), R0 = img_size/16
  └─ stage 1: depth0 x EfficientViTBlock            @ resolution R0
  └─ stage 2: Subsample(C0->C1) then depth1 blocks  @ resolution R1 = (R0-1)//2 + 1
  └─ stage 3: Subsample(C1->C2) then depth2 blocks  @ resolution R2 = (R1-1)//2 + 1
  └─ global average pool -> flatten                 -> (B, C2)
  └─ BNLinear head                                  -> (B, num_classes)
```

For the default ``img_size=224`` the stage resolutions are ``14 -> 7 -> 4``.

Fixed input size
----------------
EfficientViT's attention uses per-window relative-position bias tables built for
a specific resolution (see :mod:`models.efficientvit_block`). The model is
therefore tied to the ``img_size`` it was constructed with: ``forward``
validates that the spatial size matches. EfficientViT is **not** safe for
multi-scale / variable-resolution training; the model registry flags it
accordingly.

Example
-------
>>> import torch
>>> from models.efficientvit import efficientvit_m0
>>>
>>> model = efficientvit_m0(num_classes=1000)
>>> x = torch.randn(2, 3, 224, 224)
>>> model(x).shape
torch.Size([2, 1000])
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .common import ensure_positive_int
from .configs import (
    EfficientViTConfig,
    efficientvit_m0_config,
    efficientvit_m1_config,
    efficientvit_m2_config,
    efficientvit_m3_config,
    efficientvit_m4_config,
    efficientvit_m5_config,
)
from .efficientvit_block import EfficientViTBlock
from .efficientvit_layers import EfficientViTSubsample, PatchEmbed
from .layers import BNLinear


__all__ = [
    "EfficientViT",
    "efficientvit_m0",
    "efficientvit_m1",
    "efficientvit_m2",
    "efficientvit_m3",
    "efficientvit_m4",
    "efficientvit_m5",
]


class EfficientViT(nn.Module):
    """
    EfficientViT classification model (three hierarchical stages).

    Parameters
    ----------
    config:
        Architecture configuration (see :class:`models.configs.EfficientViTConfig`).
        Defines image size, per-stage widths/depths/heads, window size, and the
        per-head depthwise kernel sizes.

    num_classes:
        Number of output classes. Pass ``0`` to build the model headless
        (``forward`` then returns pooled features).

    in_channels:
        Number of input image channels. Default 3.

    distillation:
        If True, add a second (distillation) classifier head, DeiT-style. During
        training ``forward`` returns a tuple ``(logits, dist_logits)``; at eval it
        returns their average. Default False (single head, returns one tensor).
    """

    def __init__(
        self,
        config: EfficientViTConfig,
        num_classes: int = 1000,
        in_channels: int = 3,
        distillation: bool = False,
    ) -> None:
        super().__init__()
        ensure_positive_int(in_channels, "in_channels")
        if num_classes < 0:
            raise ValueError("num_classes must be non-negative (0 builds a headless model).")

        self.config = config
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.distillation = distillation
        self.img_size = config.img_size
        # Width of the pooled feature vector entering the head (unified contract).
        self.num_features = config.embed_dim[-1]

        embed_dim = config.embed_dim
        # Per-stage value-to-key ratio: attn_ratio_i = C_i / (key_dim_i * heads_i).
        # With EfficientViT's key dims this makes the per-head value width equal
        # the per-head channel slice, so the output projection lines up with C_i.
        attn_ratio = tuple(
            embed_dim[i] / (config.key_dim[i] * config.num_heads[i])
            for i in range(len(embed_dim))
        )

        # Stage input resolutions. R0 = img_size / patch_size; each Subsample
        # block (a 3x3 stride-2 conv) maps R -> (R - 1) // 2 + 1.
        resolution = config.img_size // config.patch_size
        stage_resolutions = [resolution]
        for _ in range(len(embed_dim) - 1):
            resolution = (resolution - 1) // 2 + 1
            stage_resolutions.append(resolution)

        # Patch-embedding stem: img_size -> img_size / 16, channels -> C0.
        self.patch_embed = PatchEmbed(in_channels, embed_dim[0])

        # Build the three stages. Stage 0 has no preceding subsample; stages 1
        # and 2 begin with an EfficientViTSubsample that halves the resolution
        # and widens the channels from the previous stage.
        stages: list[nn.Module] = []
        for stage_idx in range(len(embed_dim)):
            blocks: list[nn.Module] = []

            if stage_idx > 0:
                # Inter-stage transition: C_{i-1} -> C_i, resolution halved.
                blocks.append(
                    EfficientViTSubsample(embed_dim[stage_idx - 1], embed_dim[stage_idx])
                )

            stage_res = stage_resolutions[stage_idx]
            for _ in range(config.depth[stage_idx]):
                blocks.append(
                    EfficientViTBlock(
                        embed_dim=embed_dim[stage_idx],
                        key_dim=config.key_dim[stage_idx],
                        num_heads=config.num_heads[stage_idx],
                        attn_ratio=attn_ratio[stage_idx],
                        resolution=stage_res,
                        window_resolution=config.window_size[stage_idx],
                        kernels=config.kernels,
                    )
                )
            stages.append(nn.Sequential(*blocks))

        # Named for readability and to mirror the reference checkpoints.
        self.stages = nn.ModuleList(stages)

        # Classification head(s). BNLinear normalises the pooled vector before
        # the final Linear (the LeViT/EfficientViT convention).
        self.head = (
            BNLinear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        )
        if distillation:
            self.head_dist = (
                BNLinear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
            )

        # NOTE: weights are initialised inside ConvBN / BNLinear constructors
        # (including the residual-friendly zero-BN init and trunc-normal head),
        # so we deliberately do NOT run a global re-initialisation here.

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        """
        Return parameter names that should be excluded from weight decay.

        The learned relative-position bias tables (``attention_biases``) behave
        like positional encodings and are conventionally trained without weight
        decay.
        """
        return {name for name in self.state_dict() if "attention_biases" in name}

    def reset_classifier(self, num_classes: int) -> None:
        """
        Replace the classification head(s) with fresh ``BNLinear`` layers.

        Parameters
        ----------
        num_classes:
            New number of output classes. Pass ``<= 0`` to drop the head
            (``nn.Identity``), making ``forward`` return pooled features.
        """
        self.num_classes = num_classes
        if num_classes <= 0:
            self.head = nn.Identity()
            if self.distillation:
                self.head_dist = nn.Identity()
            return
        self.head = BNLinear(self.num_features, num_classes)
        if self.distillation:
            self.head_dist = BNLinear(self.num_features, num_classes)

    def forward_features(self, x: Tensor) -> Tensor:
        """
        Run the stem and all stages, returning the final spatial feature map.

        Input shape:  (B, in_channels, img_size, img_size)
        Output shape: (B, embed_dim[-1], R2, R2)   (R2 = img_size/16/2/2 region)
        """
        x = self.patch_embed(x)
        for stage in self.stages:
            x = stage(x)
        return x

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """
        Run the full EfficientViT model.

        Input shape:  (B, in_channels, img_size, img_size)
        Output shape: (B, num_classes)            (single head)
                      tuple of two such tensors    (distillation, training only)
        """
        if x.ndim != 4:
            raise ValueError(
                f"EfficientViT expects input shape (B, C, H, W), got {tuple(x.shape)}."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"EfficientViT was built for in_channels={self.in_channels}, "
                f"got {x.shape[1]}."
            )
        if x.shape[2] != self.img_size or x.shape[3] != self.img_size:
            raise ValueError(
                "EfficientViT is a fixed-resolution model: expected input of size "
                f"{(self.img_size, self.img_size)}, got {tuple(x.shape[2:])}. "
                "Rebuild the model with the desired img_size."
            )

        # Backbone -> (B, C2, R2, R2), then global average pool -> (B, C2).
        x = self.forward_features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)

        if self.distillation:
            out = self.head(x), self.head_dist(x)
            # At inference, average the two heads into a single prediction.
            if not self.training:
                return (out[0] + out[1]) / 2
            return out
        return self.head(x)


def _build_efficientvit(
    config: EfficientViTConfig,
    num_classes: int,
    **kwargs: Any,
) -> EfficientViT:
    """Shared constructor used by the M0..M5 factory functions."""
    return EfficientViT(config=config, num_classes=num_classes, **kwargs)


def efficientvit_m0(num_classes: int = 1000, **kwargs: Any) -> EfficientViT:
    """Build EfficientViT-M0 (smallest; embed_dim 64/128/192)."""
    return _build_efficientvit(efficientvit_m0_config(), num_classes, **kwargs)


def efficientvit_m1(num_classes: int = 1000, **kwargs: Any) -> EfficientViT:
    """Build EfficientViT-M1 (embed_dim 128/144/192)."""
    return _build_efficientvit(efficientvit_m1_config(), num_classes, **kwargs)


def efficientvit_m2(num_classes: int = 1000, **kwargs: Any) -> EfficientViT:
    """Build EfficientViT-M2 (embed_dim 128/192/224)."""
    return _build_efficientvit(efficientvit_m2_config(), num_classes, **kwargs)


def efficientvit_m3(num_classes: int = 1000, **kwargs: Any) -> EfficientViT:
    """Build EfficientViT-M3 (embed_dim 128/240/320)."""
    return _build_efficientvit(efficientvit_m3_config(), num_classes, **kwargs)


def efficientvit_m4(num_classes: int = 1000, **kwargs: Any) -> EfficientViT:
    """Build EfficientViT-M4 (embed_dim 128/256/384)."""
    return _build_efficientvit(efficientvit_m4_config(), num_classes, **kwargs)


def efficientvit_m5(num_classes: int = 1000, **kwargs: Any) -> EfficientViT:
    """Build EfficientViT-M5 (largest; embed_dim 192/288/384)."""
    return _build_efficientvit(efficientvit_m5_config(), num_classes, **kwargs)
