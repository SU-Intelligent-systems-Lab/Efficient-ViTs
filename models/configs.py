"""
Architecture configuration dataclasses and per-variant factory functions for
every model family in this library.

The library follows a config-driven design: a model class never hard-codes its
own variants. Instead a frozen dataclass describes the architecture, and a
factory function returns the dataclass for each named variant. This keeps the
"numbers" (widths, depths, head counts) separate from the "wiring" (the
``nn.Module`` graph), so a new variant is one short factory function.

Three families live here:

- MobileViT (``MobileViTConfig``):       a flat list of backbone blocks.
- EfficientViT (``EfficientViTConfig``): per-stage widths/depths/heads.
- EfficientFormer (``EfficientFormerConfig``): per-stage widths/depths + the
  number of trailing 3D attention blocks.

Every config is a ``@dataclass(frozen=True)`` (immutable and hashable) using
``tuple`` rather than ``list`` for sequence fields so the instances cannot be
mutated by accident, and validates all of its fields in ``__post_init__`` so a
bad configuration fails loudly at construction time instead of deep inside a
forward pass.

Example
-------
>>> from models.configs import mobilevit_s_config, efficientvit_m0_config
>>> mobilevit_s_config().final_conv_channels
640
>>> efficientvit_m0_config().embed_dim
(64, 128, 192)
"""

from __future__ import annotations

from dataclasses import dataclass


__all__ = [
    # MobileViT
    "MobileNetV2BlockConfig",
    "MobileViTBlockConfig",
    "BackboneBlockConfig",
    "MobileViTConfig",
    "mobilevit_s_config",
    "mobilevit_xs_config",
    "mobilevit_xxs_config",
    # EfficientViT
    "EfficientViTConfig",
    "efficientvit_m0_config",
    "efficientvit_m1_config",
    "efficientvit_m2_config",
    "efficientvit_m3_config",
    "efficientvit_m4_config",
    "efficientvit_m5_config",
    # EfficientFormer
    "EfficientFormerConfig",
    "efficientformer_l1_config",
    "efficientformer_l3_config",
    "efficientformer_l7_config",
    "efficientformer_l3_mini_config",
    "efficientformer_l7_mini_config",
]


# =============================================================================
# MobileViT
# =============================================================================


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
    Downsampling, when required between blocks, is performed by a preceding
    MobileNetV2 block (with stride 2).

    Parameters
    ----------
    out_channels:
        Number of output channels for the block.

    transformer_dim:
        Channel dimension used inside the Transformer encoder of the block.

    transformer_depth:
        Number of Transformer encoder blocks (L in the paper).
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


# Either kind of block may appear in the backbone sequence.
BackboneBlockConfig = MobileNetV2BlockConfig | MobileViTBlockConfig


@dataclass(frozen=True)
class MobileViTConfig:
    """
    Full configuration for a MobileViT model.

    The backbone is a flat sequence of block configurations. Each entry is
    exactly one block, in the order in which it is applied to the feature
    map.

    Parameters
    ----------
    stem_channels:
        Number of output channels of the initial 3 x 3 stride-2 convolution.

    blocks:
        Flat sequence of MobileNetV2BlockConfig and MobileViTBlockConfig
        objects, one entry per backbone block.

    final_conv_channels:
        Number of output channels of the 1 x 1 convolution applied after
        the backbone and before the classifier.

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


def mobilevit_s_config() -> MobileViTConfig:
    """
    Return the standard MobileViT-S configuration from the paper
    (approximately 5.6M parameters).
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
    Return the standard MobileViT-XS configuration from the paper
    (approximately 2.3M parameters).
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
    Return the standard MobileViT-XXS configuration from the paper
    (approximately 1.3M parameters).

    Note: XXS uses a smaller MV2 expansion ratio (2.0) to keep the
    parameter count low.
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


# =============================================================================
# EfficientViT (Liu et al., CVPR 2023)
# =============================================================================


@dataclass(frozen=True)
class EfficientViTConfig:
    """
    Configuration for an EfficientViT model (three hierarchical stages).

    All sequence fields are 3-tuples, one entry per stage. The same
    ``key_dim`` and ``window_size`` are used by every published M0..M5 variant,
    so only ``embed_dim``, ``depth``, ``num_heads`` and ``kernels`` differ.

    Parameters
    ----------
    img_size:
        Square input image side length. EfficientViT is fixed-resolution.

    patch_size:
        Effective patch size of the conv stem (always 16: four stride-2 convs).

    embed_dim:
        Channel width at each of the three stages.

    key_dim:
        Per-head query/key width at each stage.

    depth:
        Number of EfficientViT blocks at each stage.

    num_heads:
        Number of cascaded attention heads at each stage. ``embed_dim[i]`` must
        be divisible by ``num_heads[i]``.

    window_size:
        Local attention window side length at each stage.

    kernels:
        Per-head depthwise kernel sizes for the query enhancement. Must provide
        at least ``max(num_heads)`` entries.
    """

    img_size: int = 224
    patch_size: int = 16
    embed_dim: tuple[int, int, int] = (64, 128, 192)
    key_dim: tuple[int, int, int] = (16, 16, 16)
    depth: tuple[int, int, int] = (1, 2, 3)
    num_heads: tuple[int, int, int] = (4, 4, 4)
    window_size: tuple[int, int, int] = (7, 7, 7)
    kernels: tuple[int, ...] = (5, 5, 5, 5)

    def __post_init__(self) -> None:
        if self.img_size <= 0:
            raise ValueError("img_size must be a positive integer.")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be a positive integer.")
        if self.img_size % self.patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size.")

        for name in ("embed_dim", "key_dim", "depth", "num_heads", "window_size"):
            value = getattr(self, name)
            if len(value) != 3:
                raise ValueError(f"{name} must have exactly 3 entries (one per stage).")
            if any(v <= 0 for v in value):
                raise ValueError(f"all {name} entries must be positive integers.")

        if self.embed_dim[0] % 8 != 0:
            raise ValueError(
                "embed_dim[0] must be divisible by 8 (the conv stem uses "
                "widths embed_dim[0]/8, /4, /2)."
            )

        for stage, (dim, heads) in enumerate(zip(self.embed_dim, self.num_heads)):
            if dim % heads != 0:
                raise ValueError(
                    f"embed_dim[{stage}]={dim} must be divisible by "
                    f"num_heads[{stage}]={heads}."
                )

        if len(self.kernels) < max(self.num_heads):
            raise ValueError(
                f"kernels must provide at least max(num_heads)={max(self.num_heads)} "
                f"entries, got {len(self.kernels)}."
            )


def _efficientvit_config(**overrides: object) -> EfficientViTConfig:
    """
    Build an EfficientViTConfig from the shared defaults plus per-variant
    overrides. ``key_dim``, ``window_size``, ``img_size`` and ``patch_size`` are
    identical across all six published variants, so only the differing fields
    are overridden by the factories below.
    """
    return EfficientViTConfig(**overrides)  # type: ignore[arg-type]


def efficientvit_m0_config() -> EfficientViTConfig:
    """EfficientViT-M0: embed 64/128/192, depth 1/2/3, heads 4/4/4."""
    return _efficientvit_config(
        embed_dim=(64, 128, 192),
        depth=(1, 2, 3),
        num_heads=(4, 4, 4),
        kernels=(5, 5, 5, 5),
    )


def efficientvit_m1_config() -> EfficientViTConfig:
    """EfficientViT-M1: embed 128/144/192, depth 1/2/3, heads 2/3/3."""
    return _efficientvit_config(
        embed_dim=(128, 144, 192),
        depth=(1, 2, 3),
        num_heads=(2, 3, 3),
        kernels=(7, 5, 3, 3),
    )


def efficientvit_m2_config() -> EfficientViTConfig:
    """EfficientViT-M2: embed 128/192/224, depth 1/2/3, heads 4/3/2."""
    return _efficientvit_config(
        embed_dim=(128, 192, 224),
        depth=(1, 2, 3),
        num_heads=(4, 3, 2),
        kernels=(7, 5, 3, 3),
    )


def efficientvit_m3_config() -> EfficientViTConfig:
    """EfficientViT-M3: embed 128/240/320, depth 1/2/3, heads 4/3/4."""
    return _efficientvit_config(
        embed_dim=(128, 240, 320),
        depth=(1, 2, 3),
        num_heads=(4, 3, 4),
        kernels=(5, 5, 5, 5),
    )


def efficientvit_m4_config() -> EfficientViTConfig:
    """EfficientViT-M4: embed 128/256/384, depth 1/2/3, heads 4/4/4."""
    return _efficientvit_config(
        embed_dim=(128, 256, 384),
        depth=(1, 2, 3),
        num_heads=(4, 4, 4),
        kernels=(7, 5, 3, 3),
    )


def efficientvit_m5_config() -> EfficientViTConfig:
    """EfficientViT-M5: embed 192/288/384, depth 1/3/4, heads 3/3/4."""
    return _efficientvit_config(
        embed_dim=(192, 288, 384),
        depth=(1, 3, 4),
        num_heads=(3, 3, 4),
        kernels=(7, 5, 3, 3),
    )


# =============================================================================
# EfficientFormer (Li et al., NeurIPS 2022)
# =============================================================================


@dataclass(frozen=True)
class EfficientFormerConfig:
    """
    Configuration for an EfficientFormer model (four hierarchical stages).

    Parameters
    ----------
    name:
        Human-readable variant name (e.g. "L1").

    embed_dims:
        Channel width at each of the four stages.

    depths:
        Number of MetaBlocks at each stage.

    num_mb3d:
        Number of MB3D (3D attention) blocks at the *end* of stage 4. The
        remaining stage-4 blocks, and all earlier stages, are MB4D. Must be in
        ``[0, depths[3]]``.

    mlp_ratio:
        MLP expansion ratio used by every block. Default 4.0 (paper value).

    num_heads:
        Number of attention heads in MB3D blocks. ``embed_dims[3]`` must be
        divisible by this. Default 8 (paper value).

    qk_dim:
        Per-head query/key dimension in MB3D attention. Default 32 (paper value).

    drop_path_rate:
        Maximum stochastic-depth rate; per-block rates increase linearly from 0
        to this value across the whole network.
    """

    name: str
    embed_dims: tuple[int, int, int, int]
    depths: tuple[int, int, int, int]
    num_mb3d: int
    mlp_ratio: float = 4.0
    num_heads: int = 8
    qk_dim: int = 32
    drop_path_rate: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be a non-empty string.")

        for field_name in ("embed_dims", "depths"):
            value = getattr(self, field_name)
            if len(value) != 4:
                raise ValueError(f"{field_name} must have exactly 4 entries (one per stage).")
            if any(v <= 0 for v in value):
                raise ValueError(f"all {field_name} entries must be positive integers.")

        if not 0 <= self.num_mb3d <= self.depths[3]:
            raise ValueError(
                f"num_mb3d ({self.num_mb3d}) must be between 0 and the stage-4 "
                f"depth ({self.depths[3]})."
            )

        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive.")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be a positive integer.")
        if self.qk_dim <= 0:
            raise ValueError("qk_dim must be a positive integer.")
        if self.embed_dims[3] % self.num_heads != 0:
            raise ValueError(
                f"embed_dims[3]={self.embed_dims[3]} must be divisible by "
                f"num_heads={self.num_heads} (MB3D attention runs in stage 4)."
            )
        if not 0.0 <= self.drop_path_rate <= 1.0:
            raise ValueError("drop_path_rate must be between 0 and 1.")


def efficientformer_l1_config() -> EfficientFormerConfig:
    """
    EfficientFormer-L1 (paper Table 6, exact). ~12M params.
    Stage 4 has 4 blocks: 3 MB4D + 1 MB3D.
    """
    return EfficientFormerConfig(
        name="L1",
        embed_dims=(48, 96, 224, 448),
        depths=(3, 2, 6, 4),
        num_mb3d=1,
        drop_path_rate=0.0,
    )


def efficientformer_l3_config() -> EfficientFormerConfig:
    """
    EfficientFormer-L3 (paper-exact). ~31M params.
    Stage 4 has 6 blocks: 2 MB4D + 4 MB3D.
    """
    return EfficientFormerConfig(
        name="L3",
        embed_dims=(64, 128, 320, 512),
        depths=(4, 4, 12, 6),
        num_mb3d=4,
        drop_path_rate=0.1,
    )


def efficientformer_l7_config() -> EfficientFormerConfig:
    """
    EfficientFormer-L7 (paper-exact). ~82M params.
    Stage 4 has 8 blocks, all MB3D.
    """
    return EfficientFormerConfig(
        name="L7",
        embed_dims=(96, 192, 384, 768),
        depths=(6, 6, 8, 8),
        num_mb3d=8,
        drop_path_rate=0.2,
    )


def efficientformer_l3_mini_config() -> EfficientFormerConfig:
    """
    EfficientFormer-L3-mini: depth-reduced L3 for tractable small-dataset
    training (keeps L3's stage widths). ~15M params.
    """
    return EfficientFormerConfig(
        name="L3-mini",
        embed_dims=(64, 128, 320, 512),
        depths=(3, 3, 6, 4),
        num_mb3d=2,
        drop_path_rate=0.05,
    )


def efficientformer_l7_mini_config() -> EfficientFormerConfig:
    """
    EfficientFormer-L7-mini: depth-reduced L7 for tractable small-dataset
    training (keeps L7's wide stage widths). ~35M params.
    """
    return EfficientFormerConfig(
        name="L7-mini",
        embed_dims=(96, 192, 384, 768),
        depths=(4, 4, 4, 4),
        num_mb3d=4,
        drop_path_rate=0.1,
    )
