"""
Forward-shape and unified-contract tests for every model family.

Includes a lightweight build+forward smoke test for *every* registered variant
(at small input sizes), plus per-family contract checks for
``forward`` / ``forward_features`` / ``reset_classifier`` / ``num_features``.
"""

import pytest
import torch

from models import create_model, get_model_info, list_models


def smoke_size(name: str) -> int:
    """Pick a small but valid input size for a model, by family.

    EfficientFormer needs a multiple of 32; EfficientViT a multiple of 16;
    dynamic models accept anything.
    """
    family = get_model_info(name).family
    if family == "efficientformer":
        return 32
    if family == "efficientvit":
        return 64
    return 64


# One representative model per family for the detailed contract checks.
REPRESENTATIVES = [
    "mobilevit_xs",
    "efficientvit_m0",
    "efficientformer_l1",
    "torchvision_mobilenet_v2",
]


@pytest.mark.parametrize("name", REPRESENTATIVES)
def test_unified_contract(name):
    size = smoke_size(name)
    model = create_model(name, num_classes=8, image_size=size).eval()

    # num_features is a positive int.
    assert isinstance(model.num_features, int) and model.num_features > 0

    x = torch.randn(2, 3, size, size)
    with torch.no_grad():
        logits = model(x)
        feats = model.forward_features(x)

    # forward -> (B, num_classes); forward_features -> a tensor.
    assert logits.shape == (2, 8)
    assert torch.is_tensor(feats)

    # reset_classifier retargets the head.
    model.reset_classifier(3)
    with torch.no_grad():
        logits2 = model(x)
    assert logits2.shape == (2, 3)


@pytest.mark.parametrize("name", list_models())
def test_every_variant_forward_smoke(name):
    """Build and run a forward pass for every registered variant."""
    size = smoke_size(name)
    model = create_model(name, num_classes=5, image_size=size).eval()
    x = torch.randn(1, 3, size, size)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 5)


@pytest.mark.parametrize(
    "name,build_size,bad_size",
    [
        ("efficientvit_m0", 64, 96),
        ("efficientformer_l1", 32, 64),
    ],
)
def test_fixed_resolution_models_reject_wrong_size(name, build_size, bad_size):
    model = create_model(name, num_classes=5, image_size=build_size).eval()
    # Correct size works...
    with torch.no_grad():
        model(torch.randn(1, 3, build_size, build_size))
    # ...wrong size raises a clear error.
    with pytest.raises(ValueError):
        model(torch.randn(1, 3, bad_size, bad_size))


def test_mobilevit_accepts_variable_sizes():
    """A dynamic model runs at several resolutions without rebuilding."""
    model = create_model("mobilevit_xxs", num_classes=5).eval()
    for size in (64, 96, 128):
        with torch.no_grad():
            out = model(torch.randn(1, 3, size, size))
        assert out.shape == (1, 5)


def test_forward_rejects_wrong_rank():
    model = create_model("mobilevit_xs", num_classes=5).eval()
    with pytest.raises(ValueError):
        model(torch.randn(3, 64, 64))  # missing batch/channel split
