"""
Model components and architectures for the Light-weight ViTs library.

This package merges three model families into one consistent, bottom-up,
heavily-documented codebase:

- **MobileViT** (Mehta & Rastegari, ICLR 2022) -- hybrid CNN/Transformer.
- **EfficientViT** (Liu et al., CVPR 2023) -- cascaded group attention.
- **EfficientFormer** (Li et al., NeurIPS 2022) -- dimension-consistent design.

plus clean wrappers around torchvision CNN baselines (ShuffleNetV2,
EfficientNet, MobileNetV2/V3).

Layered structure (bottom-up)
-----------------------------
- ``common``               : validation, parameter counting, weight init.
- ``stochastic``           : DropPath and the Residual wrapper.
- ``layers``               : ConvBN, ConvNormAct, BNLinear, SqueezeExcite.
- ``mobilenetv2``/``transformer``/``mobilevit_block`` : MobileViT primitives.
- ``efficientvit_layers``/``efficientvit_block``      : EfficientViT primitives.
- ``efficientformer_block``                           : EfficientFormer primitives.
- ``mobilevit``/``efficientvit``/``efficientformer``  : top-level models + factories.
- ``torchvision_models``   : torchvision builders with the unified contract.
- ``configs``              : all architecture configs + per-variant factories.
- ``registry``             : ``create_model`` / ``list_models`` over everything.

Unified model contract
-----------------------
Every model exposes ``forward(x) -> (B, num_classes)``,
``forward_features(x)``, ``reset_classifier(num_classes)`` and a
``num_features`` attribute, so they are interchangeable in the trainer,
predictor, and benchmark.

Example
-------
>>> from models import create_model, list_models
>>> len(list_models()) > 20
True
>>> model = create_model("mobilevit_xs", num_classes=100)
"""

from __future__ import annotations

# --- shared foundations -----------------------------------------------------
from .common import (
    count_parameters,
    count_parameters_millions,
    ensure_positive_float,
    ensure_positive_int,
    ensure_probability,
    init_module_weights,
    make_divisible,
)
from .stochastic import DropPath, Residual, drop_path
from .layers import BNLinear, ConvBN, ConvNormAct, SqueezeExcite

# --- MobileViT --------------------------------------------------------------
from .mobilenetv2 import DepthwiseConv, MobileNetV2Block, PointwiseConv
from .transformer import EncoderBlock, MLP, SelfAttention, TransformerEncoder
from .mobilevit_block import MobileViTBlock
from .mobilevit import MobileViT, mobilevit_s, mobilevit_xs, mobilevit_xxs

# --- EfficientViT -----------------------------------------------------------
from .efficientvit_layers import (
    EfficientViTSubsample,
    FFN,
    PatchEmbed,
    PatchMerging,
)
from .efficientvit_block import (
    CascadedGroupAttention,
    EfficientViTBlock,
    LocalWindowAttention,
)
from .efficientvit import (
    EfficientViT,
    efficientvit_m0,
    efficientvit_m1,
    efficientvit_m2,
    efficientvit_m3,
    efficientvit_m4,
    efficientvit_m5,
)

# --- EfficientFormer --------------------------------------------------------
from .efficientformer_block import (
    ConvMLP,
    ConvStem,
    EfficientFormerAttention,
    LinearMLP,
    MB3D,
    MB4D,
    PoolMixer,
    StageDownsample,
)
from .efficientformer import (
    EfficientFormer,
    efficientformer_l1,
    efficientformer_l3,
    efficientformer_l3_mini,
    efficientformer_l7,
    efficientformer_l7_mini,
)

# --- torchvision wrappers ---------------------------------------------------
from .torchvision_models import (
    TorchvisionModel,
    build_torchvision_model,
    list_torchvision_models,
)

# --- configs ----------------------------------------------------------------
from .configs import (
    BackboneBlockConfig,
    EfficientFormerConfig,
    EfficientViTConfig,
    MobileNetV2BlockConfig,
    MobileViTBlockConfig,
    MobileViTConfig,
    efficientformer_l1_config,
    efficientformer_l3_config,
    efficientformer_l3_mini_config,
    efficientformer_l7_config,
    efficientformer_l7_mini_config,
    efficientvit_m0_config,
    efficientvit_m1_config,
    efficientvit_m2_config,
    efficientvit_m3_config,
    efficientvit_m4_config,
    efficientvit_m5_config,
    mobilevit_s_config,
    mobilevit_xs_config,
    mobilevit_xxs_config,
)

# --- registry ---------------------------------------------------------------
from .registry import (
    ModelInfo,
    create_model,
    get_model_info,
    has_pretrained,
    list_families,
    list_models,
    list_pretrained,
    supports_dynamic_input,
)


__all__ = [
    # foundations
    "count_parameters",
    "count_parameters_millions",
    "ensure_positive_float",
    "ensure_positive_int",
    "ensure_probability",
    "init_module_weights",
    "make_divisible",
    "DropPath",
    "Residual",
    "drop_path",
    "BNLinear",
    "ConvBN",
    "ConvNormAct",
    "SqueezeExcite",
    # MobileViT
    "DepthwiseConv",
    "MobileNetV2Block",
    "PointwiseConv",
    "EncoderBlock",
    "MLP",
    "SelfAttention",
    "TransformerEncoder",
    "MobileViTBlock",
    "MobileViT",
    "mobilevit_s",
    "mobilevit_xs",
    "mobilevit_xxs",
    # EfficientViT
    "EfficientViTSubsample",
    "FFN",
    "PatchEmbed",
    "PatchMerging",
    "CascadedGroupAttention",
    "EfficientViTBlock",
    "LocalWindowAttention",
    "EfficientViT",
    "efficientvit_m0",
    "efficientvit_m1",
    "efficientvit_m2",
    "efficientvit_m3",
    "efficientvit_m4",
    "efficientvit_m5",
    # EfficientFormer
    "ConvMLP",
    "ConvStem",
    "EfficientFormerAttention",
    "LinearMLP",
    "MB3D",
    "MB4D",
    "PoolMixer",
    "StageDownsample",
    "EfficientFormer",
    "efficientformer_l1",
    "efficientformer_l3",
    "efficientformer_l3_mini",
    "efficientformer_l7",
    "efficientformer_l7_mini",
    # torchvision
    "TorchvisionModel",
    "build_torchvision_model",
    "list_torchvision_models",
    # configs
    "BackboneBlockConfig",
    "EfficientFormerConfig",
    "EfficientViTConfig",
    "MobileNetV2BlockConfig",
    "MobileViTBlockConfig",
    "MobileViTConfig",
    "efficientformer_l1_config",
    "efficientformer_l3_config",
    "efficientformer_l3_mini_config",
    "efficientformer_l7_config",
    "efficientformer_l7_mini_config",
    "efficientvit_m0_config",
    "efficientvit_m1_config",
    "efficientvit_m2_config",
    "efficientvit_m3_config",
    "efficientvit_m4_config",
    "efficientvit_m5_config",
    "mobilevit_s_config",
    "mobilevit_xs_config",
    "mobilevit_xxs_config",
    # registry
    "ModelInfo",
    "create_model",
    "get_model_info",
    "list_families",
    "list_models",
    "supports_dynamic_input",
    "list_pretrained",
    "has_pretrained",
]
