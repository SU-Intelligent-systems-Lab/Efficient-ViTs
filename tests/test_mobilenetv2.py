import pytest
import torch
from torch import nn

from models.mobilenetv2 import (
    PointwiseConv,
    DepthwiseConv,
    MobileNetV2Block,
)


def test_pointwise_conv_output_shape():
    x = torch.randn(2, 32, 16, 16)
    layer = PointwiseConv(in_channels=32, out_channels=64)

    y = layer(x)

    assert y.shape == (2, 64, 16, 16)


def test_pointwise_conv_linear_has_no_activation():
    layer = PointwiseConv(in_channels=32, out_channels=64, linear=True)

    has_activation = any(
        isinstance(module, (nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU))
        for module in layer.modules()
    )

    assert not has_activation


def test_depthwise_conv_output_shape_stride_1():
    x = torch.randn(2, 32, 16, 16)
    layer = DepthwiseConv(channels=32, stride=1)

    y = layer(x)

    assert y.shape == (2, 32, 16, 16)


def test_depthwise_conv_output_shape_stride_2():
    x = torch.randn(2, 32, 16, 16)
    layer = DepthwiseConv(channels=32, stride=2)

    y = layer(x)

    assert y.shape == (2, 32, 8, 8)


def test_depthwise_conv_uses_depthwise_groups():
    layer = DepthwiseConv(channels=32)

    conv = next(module for module in layer.modules() if isinstance(module, nn.Conv2d))

    assert conv.groups == 32
    assert conv.in_channels == 32
    assert conv.out_channels == 32


def test_mobilenetv2_block_same_shape_with_residual():
    x = torch.randn(2, 32, 16, 16)
    block = MobileNetV2Block(in_channels=32, out_channels=32, stride=1)

    y = block(x)

    assert y.shape == (2, 32, 16, 16)
    assert block.use_residual is True


def test_mobilenetv2_block_downsamples_without_residual():
    x = torch.randn(2, 32, 16, 16)
    block = MobileNetV2Block(in_channels=32, out_channels=64, stride=2)

    y = block(x)

    assert y.shape == (2, 64, 8, 8)
    assert block.use_residual is False


def test_mobilenetv2_block_changes_channels_without_residual():
    x = torch.randn(2, 32, 16, 16)
    block = MobileNetV2Block(in_channels=32, out_channels=64, stride=1)

    y = block(x)

    assert y.shape == (2, 64, 16, 16)
    assert block.use_residual is False


def test_mobilenetv2_block_expand_ratio_one():
    x = torch.randn(2, 32, 16, 16)
    block = MobileNetV2Block(
        in_channels=32,
        out_channels=32,
        stride=1,
        expand_ratio=1.0,
    )

    y = block(x)

    assert y.shape == (2, 32, 16, 16)


def test_invalid_pointwise_channels():
    with pytest.raises(ValueError):
        PointwiseConv(in_channels=0, out_channels=32)

    with pytest.raises(ValueError):
        PointwiseConv(in_channels=32, out_channels=0)


def test_invalid_depthwise_channels():
    with pytest.raises(ValueError):
        DepthwiseConv(channels=0)


def test_invalid_depthwise_stride():
    with pytest.raises(ValueError):
        DepthwiseConv(channels=32, stride=3)


def test_invalid_mobilenetv2_block_stride():
    with pytest.raises(ValueError):
        MobileNetV2Block(in_channels=32, out_channels=32, stride=3)


def test_invalid_mobilenetv2_block_expand_ratio():
    with pytest.raises(ValueError):
        MobileNetV2Block(in_channels=32, out_channels=32, expand_ratio=0.0)
