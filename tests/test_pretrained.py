"""
Tests for local (ImageNet-100) pretrained-weight loading.

The hand-built models ship weights under ``weights/<name>.pt``. These tests are
guarded with ``skipif`` so they degrade gracefully if the (large) weight files
are not present on a given checkout.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from inference import Predictor
from models import create_model, get_model_info, has_pretrained, list_pretrained
from models.pretrained import get_pretrained_path, infer_num_classes


# One representative per hand-built family (kept small for fast tests).
REPRESENTATIVES = ["mobilevit_xxs", "efficientvit_m0", "efficientformer_l1"]


def test_list_pretrained_nonempty_and_named_by_model():
    names = list_pretrained()
    # Every listed name must be a real registry model and flagged available.
    for name in names:
        assert get_model_info(name).pretrained_available is True
        assert get_pretrained_path(name) is not None


@pytest.mark.parametrize("name", REPRESENTATIVES)
def test_pretrained_loads_and_keeps_head(name):
    if not has_pretrained(name):
        pytest.skip(f"no bundled weights for {name}")
    # num_classes=None -> keep the checkpoint's own head (ImageNet-100 -> 100).
    model = create_model(name, pretrained=True).eval()
    assert model.num_classes == 100
    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 100)


@pytest.mark.parametrize("name", REPRESENTATIVES)
def test_pretrained_can_retarget_head(name):
    if not has_pretrained(name):
        pytest.skip(f"no bundled weights for {name}")
    model = create_model(name, num_classes=10, pretrained=True).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 10)


def test_infer_num_classes_roundtrip():
    # A freshly built 100-class model's state_dict reports 100 classes.
    model = create_model("mobilevit_xxs", num_classes=100)
    assert infer_num_classes(model.state_dict()) == 100


def test_predictor_from_pretrained():
    name = "mobilevit_xxs"
    if not has_pretrained(name):
        pytest.skip(f"no bundled weights for {name}")
    predictor = Predictor.from_pretrained(name, device="cpu")
    img = Image.fromarray((np.random.rand(150, 150, 3) * 255).astype("uint8"))
    probs, idx = predictor.predict(img, topk=5)
    assert probs.shape == (5,) and idx.shape == (5,)
