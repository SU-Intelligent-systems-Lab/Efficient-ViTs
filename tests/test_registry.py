"""Tests for the unified model registry (create_model / list_models)."""

import pytest
import torch

from models import (
    create_model,
    get_model_info,
    has_pretrained,
    list_families,
    list_models,
    supports_dynamic_input,
)


def test_list_models_covers_all_families():
    names = list_models()
    # 3 MobileViT + 6 EfficientViT + 5 EfficientFormer + 18 torchvision = 32.
    assert len(names) == 32
    assert set(list_families()) == {"mobilevit", "efficientvit", "efficientformer", "torchvision"}
    assert len(list_models(family="mobilevit")) == 3
    assert len(list_models(family="efficientvit")) == 6
    assert len(list_models(family="efficientformer")) == 5
    assert len(list_models(family="torchvision")) == 18


def test_dynamic_input_flags():
    # MobileViT and torchvision are dynamic; EfficientViT/EfficientFormer are not.
    assert supports_dynamic_input("mobilevit_xs") is True
    assert supports_dynamic_input("torchvision_mobilenet_v2") is True
    assert supports_dynamic_input("efficientvit_m0") is False
    assert supports_dynamic_input("efficientformer_l1") is False

    dynamic = set(list_models(dynamic_input=True))
    fixed = set(list_models(dynamic_input=False))
    assert dynamic.isdisjoint(fixed)
    assert dynamic | fixed == set(list_models())


def test_create_model_one_per_family():
    specs = [
        ("mobilevit_xxs", 64),
        ("efficientvit_m0", 64),
        ("efficientformer_l1", 32),
        ("torchvision_shufflenet_v2_x1_0", 64),
    ]
    for name, size in specs:
        model = create_model(name, num_classes=7, image_size=size).eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, size, size))
        assert out.shape == (2, 7)


def test_create_model_unknown_raises():
    with pytest.raises(ValueError):
        create_model("does_not_exist")


def test_get_model_info_fields():
    info = get_model_info("efficientvit_m0")
    assert info.family == "efficientvit"
    assert info.dynamic_input is False
    assert info.default_image_size == 224


def test_pretrained_missing_weights_raises():
    # A hand-built model with no bundled weights raises FileNotFoundError
    # (rather than silently returning a randomly-initialised model).
    name = "efficientformer_l7"  # not trained in this library -> no weights file
    if has_pretrained(name):
        pytest.skip("weights unexpectedly present for this model")
    with pytest.raises(FileNotFoundError):
        create_model(name, pretrained=True)


def test_torchvision_rejects_non_rgb():
    with pytest.raises(ValueError):
        create_model("torchvision_mobilenet_v2", num_classes=10, in_channels=1)
