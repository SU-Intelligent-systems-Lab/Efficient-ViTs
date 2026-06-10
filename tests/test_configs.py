"""Tests for the architecture config dataclasses and their variant factories."""

import dataclasses

import pytest

from models import configs


# Every config factory and the family of config it should produce.
EFFICIENTVIT_FACTORIES = [
    configs.efficientvit_m0_config,
    configs.efficientvit_m1_config,
    configs.efficientvit_m2_config,
    configs.efficientvit_m3_config,
    configs.efficientvit_m4_config,
    configs.efficientvit_m5_config,
]
EFFICIENTFORMER_FACTORIES = [
    configs.efficientformer_l1_config,
    configs.efficientformer_l3_config,
    configs.efficientformer_l7_config,
    configs.efficientformer_l3_mini_config,
    configs.efficientformer_l7_mini_config,
]
MOBILEVIT_FACTORIES = [
    configs.mobilevit_s_config,
    configs.mobilevit_xs_config,
    configs.mobilevit_xxs_config,
]


@pytest.mark.parametrize("factory", MOBILEVIT_FACTORIES)
def test_mobilevit_config_factories(factory):
    config = factory()
    assert isinstance(config, configs.MobileViTConfig)
    # blocks must be a tuple (immutable), with 10 entries in the paper layout.
    assert isinstance(config.blocks, tuple)
    assert len(config.blocks) == 10
    assert config.final_conv_channels > 0


@pytest.mark.parametrize("factory", EFFICIENTVIT_FACTORIES)
def test_efficientvit_config_factories(factory):
    config = factory()
    assert isinstance(config, configs.EfficientViTConfig)
    # Per-stage fields are 3-tuples.
    for field in ("embed_dim", "key_dim", "depth", "num_heads", "window_size"):
        value = getattr(config, field)
        assert isinstance(value, tuple) and len(value) == 3
    # embed_dim divisible by num_heads at every stage.
    for dim, heads in zip(config.embed_dim, config.num_heads):
        assert dim % heads == 0


@pytest.mark.parametrize("factory", EFFICIENTFORMER_FACTORIES)
def test_efficientformer_config_factories(factory):
    config = factory()
    assert isinstance(config, configs.EfficientFormerConfig)
    assert isinstance(config.embed_dims, tuple) and len(config.embed_dims) == 4
    assert isinstance(config.depths, tuple) and len(config.depths) == 4
    # MB3D blocks live only at the tail of stage 4.
    assert 0 <= config.num_mb3d <= config.depths[3]


def test_configs_are_frozen():
    config = configs.mobilevit_s_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.stem_channels = 99  # type: ignore[misc]


def test_mobilevit_config_validation():
    with pytest.raises(ValueError):
        configs.MobileViTConfig(stem_channels=0, blocks=(), final_conv_channels=10)
    with pytest.raises(ValueError):
        # empty blocks
        configs.MobileViTConfig(stem_channels=16, blocks=(), final_conv_channels=320)


def test_efficientvit_config_validation():
    # embed_dim[0] not divisible by 8
    with pytest.raises(ValueError):
        configs.EfficientViTConfig(embed_dim=(60, 128, 192))
    # embed_dim not divisible by num_heads
    with pytest.raises(ValueError):
        configs.EfficientViTConfig(embed_dim=(64, 128, 192), num_heads=(5, 4, 4))
    # wrong tuple length
    with pytest.raises(ValueError):
        configs.EfficientViTConfig(depth=(1, 2))
    # img_size not divisible by patch_size
    with pytest.raises(ValueError):
        configs.EfficientViTConfig(img_size=100, patch_size=16)


def test_efficientformer_config_validation():
    # num_mb3d exceeds stage-4 depth
    with pytest.raises(ValueError):
        configs.EfficientFormerConfig(
            name="bad", embed_dims=(48, 96, 224, 448), depths=(3, 2, 6, 4), num_mb3d=5
        )
    # embed_dims[3] not divisible by num_heads
    with pytest.raises(ValueError):
        configs.EfficientFormerConfig(
            name="bad", embed_dims=(48, 96, 224, 450), depths=(3, 2, 6, 4), num_mb3d=1, num_heads=8
        )
    # wrong tuple length
    with pytest.raises(ValueError):
        configs.EfficientFormerConfig(
            name="bad", embed_dims=(48, 96, 224), depths=(3, 2, 6, 4), num_mb3d=1
        )
