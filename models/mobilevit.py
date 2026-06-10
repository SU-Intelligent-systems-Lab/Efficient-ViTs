"""
This module provides:
- MobileViT: the full MobileViT classification model.
- mobilevit_s, mobilevit_xs, mobilevit_xxs: factory functions for the three
  standard variants from the paper.

A MobileViT model is built as a flat sequence:
1. Stem: a 3 x 3 stride-2 convolution maps the input image to feature maps,
   halving the spatial resolution from (H, W) to (H/2, W/2).
2. Backbone: a flat sequence of MobileNetV2 and MobileViT blocks, applied
   in the order given by MobileViTConfig.blocks. MobileNetV2 blocks handle
   spatial downsampling (when stride == 2) and most parameter-efficient
   computation; MobileViT blocks model global context within their stage.
3. Final pointwise convolution: a 1 x 1 convolution that expands the channel
   dimension before pooling (a common ImageNet-style trick that gives the
   classifier a wider feature vector to work with).
4. Global average pooling: collapses spatial dimensions to a single vector
   per sample, of length ``final_conv_channels``.
5. Classifier: a single linear layer maps the pooled features to logits.

Unified model contract
-----------------------
Like every other model in this library, MobileViT exposes:
- ``forward_features(x)`` -> the backbone feature map ``(B, final_conv_channels, H', W')``
- ``forward(x)``          -> logits ``(B, num_classes)``
- ``reset_classifier(num_classes)`` -> swap in a fresh classifier head
- ``num_features``        -> width of the pooled feature vector

Spatial-size flow (example with input 256 x 256, all MobileViT variants)
------------------------------------------------------------------------
After the stem (stride 2):                    128 x 128
After 1st MV2 stride-2 block:                  64 x  64
After 2nd MV2 stride-2 block:                  32 x  32
After 3rd MV2 stride-2 block:                  16 x  16
After 4th MV2 stride-2 block:                   8 x   8
After global average pool:                      1 x   1

MobileViT supports flexible input sizes: its patch fold/unfold dynamically
resizes feature maps to a patch-divisible size and back (see
``MobileViTBlock``), and the head uses adaptive average pooling. It is
therefore safe to use with the multi-scale sampler.

Example
-------
>>> import torch
>>> from models.mobilevit import mobilevit_s
>>>
>>> model = mobilevit_s(num_classes=1000)
>>> x = torch.randn(2, 3, 256, 256)
>>> logits = model(x)
>>> logits.shape
torch.Size([2, 1000])
"""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from .common import init_module_weights
from .configs import (
    BackboneBlockConfig,
    MobileNetV2BlockConfig,
    MobileViTBlockConfig,
    MobileViTConfig,
    mobilevit_s_config,
    mobilevit_xs_config,
    mobilevit_xxs_config,
)
from .mobilenetv2 import MobileNetV2Block, PointwiseConv
from .mobilevit_block import MobileViTBlock


__all__ = [
    "MobileViT",
    "mobilevit_s",
    "mobilevit_xs",
    "mobilevit_xxs",
]


class MobileViT(nn.Module):
    """
    MobileViT classification model.

    Combines lightweight MobileNetV2 blocks for local feature extraction
    with MobileViT blocks for global context modeling. The backbone is a
    flat sequence of blocks described by the configuration.

    Parameters
    ----------
    config:
        Architecture configuration describing the stem, backbone blocks,
        and the final pointwise convolution.

    num_classes:
        Number of output classes for the classification head.

    in_channels:
        Number of channels in the input image. Default is 3 (RGB).

    classifier_dropout:
        Dropout probability applied to pooled features before the linear
        classifier.

    attention_dropout:
        Dropout probability applied to attention probabilities inside
        MobileViT blocks.

    projection_dropout:
        Dropout probability applied after the attention output projection.

    mlp_dropout:
        Dropout probability used inside the Transformer MLP.

    drop_path:
        Stochastic depth probability used inside Transformer encoder
        blocks. Each encoder stack interpolates from 0 to this value
        across its depth.

    norm_layer:
        Normalization layer used for convolutional layers. Default is
        nn.BatchNorm2d.

    activation_layer:
        Activation layer used for convolutional layers. Default is nn.SiLU.
    """

    def __init__(
        self,
        config: MobileViTConfig,
        num_classes: int = 1000,
        in_channels: int = 3,
        classifier_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        activation_layer: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError("num_classes must be a positive integer.")

        if in_channels <= 0:
            raise ValueError("in_channels must be a positive integer.")

        if not 0.0 <= classifier_dropout <= 1.0:
            raise ValueError("classifier_dropout must be between 0 and 1.")

        if not 0.0 <= attention_dropout <= 1.0:
            raise ValueError("attention_dropout must be between 0 and 1.")

        if not 0.0 <= projection_dropout <= 1.0:
            raise ValueError("projection_dropout must be between 0 and 1.")

        if not 0.0 <= mlp_dropout <= 1.0:
            raise ValueError("mlp_dropout must be between 0 and 1.")

        if not 0.0 <= drop_path <= 1.0:
            raise ValueError("drop_path must be between 0 and 1.")

        self.config = config
        self.num_classes = num_classes
        self.in_channels = in_channels
        # Pooled feature width that feeds the classifier (unified contract).
        self.num_features = config.final_conv_channels

        # Stem: 3 x 3 stride-2 convolution that halves the spatial size.
        # (B, in_channels, H, W) -> (B, stem_channels, H/2, W/2)
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=config.stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            norm_layer(config.stem_channels),
            activation_layer(),
        )

        # Backbone: build one nn.Module per entry in config.blocks. The
        # channel count flows from one block to the next; spatial size is
        # halved by each MV2 block whose stride is 2.
        backbone: list[nn.Module] = []
        current_channels = config.stem_channels

        for block_config in config.blocks:
            backbone.append(
                self._build_block(
                    block_config=block_config,
                    in_channels=current_channels,
                    expand_ratio=config.expand_ratio,
                    patch_size=config.patch_size,
                    n_heads=config.n_heads,
                    mlp_ratio=config.mlp_ratio,
                    attention_dropout=attention_dropout,
                    projection_dropout=projection_dropout,
                    mlp_dropout=mlp_dropout,
                    drop_path=drop_path,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )
            current_channels = block_config.out_channels

        self.backbone = nn.Sequential(*backbone)

        # Final 1 x 1 convolution: expand channels to a wider classifier
        # input. (B, current_channels, h, w) -> (B, final_conv_channels, h, w)
        self.final_conv = PointwiseConv(
            in_channels=current_channels,
            out_channels=config.final_conv_channels,
            linear=False,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        # Classification head:
        # global average pool over space -> dropout -> linear to num_classes.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier_dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.final_conv_channels, num_classes)

        self._initialize_weights()

    @staticmethod
    def _build_block(
        block_config: BackboneBlockConfig,
        in_channels: int,
        expand_ratio: float,
        patch_size: int | tuple[int, int],
        n_heads: int,
        mlp_ratio: float,
        attention_dropout: float,
        projection_dropout: float,
        mlp_dropout: float,
        drop_path: float,
        norm_layer: type[nn.Module],
        activation_layer: type[nn.Module],
    ) -> nn.Module:
        """
        Build one backbone block by dispatching on the configuration type:
        - MobileNetV2BlockConfig -> MobileNetV2Block.
        - MobileViTBlockConfig   -> MobileViTBlock.
        """
        if isinstance(block_config, MobileNetV2BlockConfig):
            return MobileNetV2Block(
                in_channels=in_channels,
                out_channels=block_config.out_channels,
                stride=block_config.stride,
                expand_ratio=expand_ratio,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )

        if isinstance(block_config, MobileViTBlockConfig):
            return MobileViTBlock(
                in_channels=in_channels,
                out_channels=block_config.out_channels,
                transformer_dim=block_config.transformer_dim,
                depth=block_config.transformer_depth,
                patch_size=patch_size,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
                mlp_dropout=mlp_dropout,
                drop_path=drop_path,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )

        raise TypeError(
            f"Unsupported backbone block configuration: {type(block_config).__name__}."
        )

    def _initialize_weights(self) -> None:
        """
        Initialize convolutional, normalization, and linear weights.

        Delegates to the shared :func:`models.common.init_module_weights`
        helper, which applies Kaiming-normal init to convolutions, identity
        init to normalization layers, and truncated-normal init to linear
        layers. This is the same scheme MobileViT used originally, now shared
        across every model family in the library.
        """
        self.apply(init_module_weights)

    def reset_classifier(self, num_classes: int) -> None:
        """
        Replace the classification head with a fresh ``nn.Linear``.

        Useful for transfer learning: load a backbone trained on one dataset,
        then call ``reset_classifier(new_num_classes)`` to retarget it. Pass
        ``num_classes <= 0`` to drop the head entirely (``nn.Identity``), which
        makes ``forward`` return pooled features.

        Parameters
        ----------
        num_classes:
            New number of output classes.
        """
        self.num_classes = num_classes
        if num_classes <= 0:
            self.classifier = nn.Identity()
            return
        self.classifier = nn.Linear(self.num_features, num_classes)
        # Re-initialise just the new head, matching the global init scheme.
        self.classifier.apply(init_module_weights)

    def forward_features(self, x: Tensor) -> Tensor:
        """
        Apply the stem, backbone, and final pointwise convolution but skip
        the classification head. Useful for transfer learning.

        Input shape:  (B, C_in, H, W)
        Output shape: (B, final_conv_channels, H', W')
        """
        x = self.stem(x)
        x = self.backbone(x)
        x = self.final_conv(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Run the full MobileViT model.

        Input shape:  (B, C_in, H, W)
        Output shape: (B, num_classes)
        """
        if x.ndim != 4:
            raise ValueError(
                f"MobileViT expects input shape (B, C, H, W), but got {tuple(x.shape)}."
            )

        # Backbone: (B, C_in, H, W) -> (B, final_conv_channels, H', W')
        x = self.forward_features(x)

        # Global average pool collapses spatial dims to 1 x 1 then flatten.
        # (B, final_conv_channels, H', W') -> (B, final_conv_channels)
        x = self.pool(x)
        x = x.flatten(1)

        # Classifier: (B, final_conv_channels) -> (B, num_classes)
        x = self.classifier_dropout(x)
        x = self.classifier(x)
        return x


def mobilevit_s(num_classes: int = 1000, **kwargs: Any) -> MobileViT:
    """
    Build the standard MobileViT-S model from the paper (~5.6M parameters).

    Additional keyword arguments are forwarded to ``MobileViT.__init__``.
    """
    return MobileViT(
        config=mobilevit_s_config(),
        num_classes=num_classes,
        **kwargs,
    )


def mobilevit_xs(num_classes: int = 1000, **kwargs: Any) -> MobileViT:
    """
    Build the standard MobileViT-XS model from the paper (~2.3M parameters).

    Additional keyword arguments are forwarded to ``MobileViT.__init__``.
    """
    return MobileViT(
        config=mobilevit_xs_config(),
        num_classes=num_classes,
        **kwargs,
    )


def mobilevit_xxs(num_classes: int = 1000, **kwargs: Any) -> MobileViT:
    """
    Build the standard MobileViT-XXS model from the paper (~1.3M parameters).

    Additional keyword arguments are forwarded to ``MobileViT.__init__``.
    """
    return MobileViT(
        config=mobilevit_xxs_config(),
        num_classes=num_classes,
        **kwargs,
    )
