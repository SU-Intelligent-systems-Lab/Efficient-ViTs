import pytest
import torch

from models.mobilevit_block import MobileViTBlock


def test_mobilevit_block_output_shape_same_channels():
    x = torch.randn(2, 64, 32, 32)
    block = MobileViTBlock(
        in_channels=64,
        out_channels=64,
        transformer_dim=96,
        depth=2,
        patch_size=2,
        n_heads=4,
    )

    y = block(x)

    assert y.shape == (2, 64, 32, 32)


def test_mobilevit_block_output_shape_different_channels():
    x = torch.randn(2, 64, 32, 32)
    block = MobileViTBlock(
        in_channels=64,
        out_channels=80,
        transformer_dim=96,
        depth=2,
        patch_size=2,
        n_heads=4,
    )

    y = block(x)

    assert y.shape == (2, 80, 32, 32)


def test_mobilevit_block_preserves_odd_spatial_size():
    x = torch.randn(2, 64, 31, 29)
    block = MobileViTBlock(
        in_channels=64,
        out_channels=64,
        transformer_dim=96,
        depth=2,
        patch_size=2,
        n_heads=4,
    )

    y = block(x)

    assert y.shape == (2, 64, 31, 29)


def test_mobilevit_block_tuple_patch_size():
    x = torch.randn(2, 64, 32, 36)
    block = MobileViTBlock(
        in_channels=64,
        out_channels=64,
        transformer_dim=96,
        depth=2,
        patch_size=(2, 3),
        n_heads=4,
    )

    y = block(x)

    assert y.shape == (2, 64, 32, 36)


def test_mobilevit_block_unfold_shape():
    x = torch.randn(2, 96, 32, 32)
    block = MobileViTBlock(
        in_channels=64,
        out_channels=64,
        transformer_dim=96,
        depth=1,
        patch_size=2,
        n_heads=4,
    )

    tokens, fold_info = block._unfold(x)

    assert tokens.shape == (8, 256, 96)
    assert fold_info == (2, 96, 32, 32)


def test_mobilevit_block_fold_restores_shape():
    x = torch.randn(2, 96, 32, 32)
    block = MobileViTBlock(
        in_channels=64,
        out_channels=64,
        transformer_dim=96,
        depth=1,
        patch_size=2,
        n_heads=4,
    )

    tokens, fold_info = block._unfold(x)
    y = block._fold(tokens, fold_info)

    assert y.shape == x.shape
    assert torch.allclose(y, x)


def test_mobilevit_block_resize_if_needed():
    x = torch.randn(2, 96, 31, 29)
    block = MobileViTBlock(
        in_channels=64,
        out_channels=64,
        transformer_dim=96,
        depth=1,
        patch_size=2,
        n_heads=4,
    )

    y = block._resize_if_needed(x)

    assert y.shape == (2, 96, 32, 30)


def test_mobilevit_block_no_resize_when_divisible():
    x = torch.randn(2, 96, 32, 32)
    block = MobileViTBlock(
        in_channels=64,
        out_channels=64,
        transformer_dim=96,
        depth=1,
        patch_size=2,
        n_heads=4,
    )

    y = block._resize_if_needed(x)

    assert y.shape == x.shape
    assert torch.equal(y, x)


def test_mobilevit_block_has_transformer_norm():
    block = MobileViTBlock(
        in_channels=64,
        out_channels=64,
        transformer_dim=96,
        depth=1,
        patch_size=2,
        n_heads=4,
    )

    assert hasattr(block, "transformer_norm")


def test_mobilevit_block_invalid_input_rank():
    x = torch.randn(2, 64, 32)
    block = MobileViTBlock(
        in_channels=64,
        out_channels=64,
        transformer_dim=96,
        depth=1,
        patch_size=2,
        n_heads=4,
    )

    with pytest.raises(ValueError):
        block(x)


def test_mobilevit_block_invalid_channels():
    with pytest.raises(ValueError):
        MobileViTBlock(
            in_channels=0,
            out_channels=64,
            transformer_dim=96,
            depth=1,
            patch_size=2,
            n_heads=4,
        )

    with pytest.raises(ValueError):
        MobileViTBlock(
            in_channels=64,
            out_channels=0,
            transformer_dim=96,
            depth=1,
            patch_size=2,
            n_heads=4,
        )


def test_mobilevit_block_invalid_transformer_dim():
    with pytest.raises(ValueError):
        MobileViTBlock(
            in_channels=64,
            out_channels=64,
            transformer_dim=0,
            depth=1,
            patch_size=2,
            n_heads=4,
        )


def test_mobilevit_block_invalid_even_conv_kernel_size():
    with pytest.raises(ValueError):
        MobileViTBlock(
            in_channels=64,
            out_channels=64,
            transformer_dim=96,
            depth=1,
            patch_size=2,
            n_heads=4,
            conv_kernel_size=2,
        )


def test_mobilevit_block_invalid_patch_size():
    with pytest.raises(ValueError):
        MobileViTBlock(
            in_channels=64,
            out_channels=64,
            transformer_dim=96,
            depth=1,
            patch_size=0,
            n_heads=4,
        )

    with pytest.raises(ValueError):
        MobileViTBlock(
            in_channels=64,
            out_channels=64,
            transformer_dim=96,
            depth=1,
            patch_size=(2, 0),
            n_heads=4,
        )

    with pytest.raises(ValueError):
        MobileViTBlock(
            in_channels=64,
            out_channels=64,
            transformer_dim=96,
            depth=1,
            patch_size=(2, 2, 2),
            n_heads=4,
        )
