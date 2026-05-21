"""
This module provides:
- MobileViT: the full MobileViT classification model.
- mobilevit_s, mobilevit_xs, mobilevit_xxs: factory functions for the three
  standard variants from the paper.

A MobileViT model is built as a flat sequence:
1. Stem: a 3 x 3 stride-2 convolution maps the input image to feature maps.
2. Backbone: a flat sequence of MobileNetV2 and MobileViT blocks, applied in
   the order given by MobileViTConfig.blocks. MobileNetV2 blocks handle
   spatial downsampling; MobileViT blocks model global context.
3. Final pointwise convolution: a 1 x 1 convolution expands the channel
   dimension before pooling.
4. Global average pooling: collapses spatial dimensions to a single vector.
5. Classifier: a linear layer maps pooled features to class logits.

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

    Combines lightweight MobileNetV2 blocks for local feature extraction with
    MobileViT blocks for global context modeling. The backbone is a flat
    sequence of blocks described by the configuration.

    Parameters
    ----------
    config:
        Architecture configuration describing the stem, backbone blocks, and
        final pointwise convolution.

    num_classes:
        Number of output classes for the classification head.

    in_channels:
        Number of channels in the input image. Default is 3.

    classifier_dropout:
        Dropout probability applied to pooled features before the linear classifier.

    attention_dropout:
        Dropout probability applied to attention probabilities inside MobileViT blocks.

    projection_dropout:
        Dropout probability applied after the attention output projection.

    mlp_dropout:
        Dropout probability used inside the Transformer MLP.

    drop_path:
        Stochastic depth probability used inside Transformer encoder blocks.

    norm_layer:
        Normalization layer used for convolutional layers. Default is nn.BatchNorm2d.

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

        # Stem: 3 x 3 stride-2 convolution that halves the spatial resolution.
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

        # Backbone: one nn.Module per entry in config.blocks.
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

        # Final pointwise expansion before the classification head.
        self.final_conv = PointwiseConv(
            in_channels=current_channels,
            out_channels=config.final_conv_channels,
            linear=False,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        # Classification head.
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
        Build one backbone block.

        Dispatches on the concrete block configuration type:
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

        Conv layers use Kaiming-normal initialization. Normalization layers
        use unit weight and zero bias. Linear layers use truncated normal
        initialization with std 0.02.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_features(self, x: Tensor) -> Tensor:
        """
        Apply the stem, backbone, and final pointwise convolution.

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

        x = self.forward_features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.classifier_dropout(x)
        x = self.classifier(x)
        return x


def mobilevit_s(num_classes: int = 1000, **kwargs: Any) -> MobileViT:
    """
    Build the standard MobileViT-S model from the paper.

    Has approximately 5.6M parameters.

    Additional keyword arguments are forwarded to MobileViT.
    """
    return MobileViT(
        config=mobilevit_s_config(),
        num_classes=num_classes,
        **kwargs,
    )


def mobilevit_xs(num_classes: int = 1000, **kwargs: Any) -> MobileViT:
    """
    Build the standard MobileViT-XS model from the paper.

    Has approximately 2.3M parameters.

    Additional keyword arguments are forwarded to MobileViT.
    """
    return MobileViT(
        config=mobilevit_xs_config(),
        num_classes=num_classes,
        **kwargs,
    )


def mobilevit_xxs(num_classes: int = 1000, **kwargs: Any) -> MobileViT:
    """
    Build the standard MobileViT-XXS model from the paper.

    Has approximately 1.3M parameters.

    Additional keyword arguments are forwarded to MobileViT.
    """
    return MobileViT(
        config=mobilevit_xxs_config(),
        num_classes=num_classes,
        **kwargs,
    )
