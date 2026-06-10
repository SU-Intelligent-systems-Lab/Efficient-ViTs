"""Tests for the inference Predictor."""

import numpy as np
import pytest
from PIL import Image

from inference import Predictor
from models import create_model
from training import save_model


def _random_image(size=96):
    arr = (np.random.rand(size, size, 3) * 255).astype("uint8")
    return Image.fromarray(arr)


def test_predictor_from_checkpoint_roundtrip(tmp_path):
    ckpt = tmp_path / "best.pt"
    model = create_model("mobilevit_xs", num_classes=6)
    save_model(ckpt, model)

    predictor = Predictor.from_checkpoint(
        ckpt,
        model_name="mobilevit_xs",
        num_classes=6,
        image_size=64,
        device="cpu",
        class_names=[f"class_{i}" for i in range(6)],
    )

    probs, indices = predictor.predict(_random_image(), topk=3)
    assert probs.shape == (3,)
    assert indices.shape == (3,)
    # Probabilities are sorted descending and valid.
    assert probs[0] >= probs[1] >= probs[2]
    assert predictor.class_name(indices[0]).startswith("class_")


def test_predictor_batch(tmp_path):
    ckpt = tmp_path / "best.pt"
    model = create_model("mobilevit_xxs", num_classes=4)
    save_model(ckpt, model)

    predictor = Predictor.from_checkpoint(
        ckpt, model_name="mobilevit_xxs", num_classes=4, image_size=64, device="cpu",
    )
    images = [_random_image(), _random_image()]
    probs, indices = predictor.predict_batch(images, topk=2)
    assert probs.shape == (2, 2)
    assert indices.shape == (2, 2)


def test_predictor_class_name_without_names(tmp_path):
    ckpt = tmp_path / "best.pt"
    model = create_model("mobilevit_xxs", num_classes=4)
    save_model(ckpt, model)
    predictor = Predictor.from_checkpoint(
        ckpt, model_name="mobilevit_xxs", num_classes=4, image_size=64, device="cpu",
    )
    # No class names -> falls back to the string index.
    assert predictor.class_name(2) == "2"


def test_predictor_topk_validation(tmp_path):
    ckpt = tmp_path / "best.pt"
    model = create_model("mobilevit_xxs", num_classes=4)
    save_model(ckpt, model)
    predictor = Predictor.from_checkpoint(
        ckpt, model_name="mobilevit_xxs", num_classes=4, image_size=64, device="cpu",
    )
    with pytest.raises(ValueError):
        predictor.predict(_random_image(), topk=0)
    with pytest.raises(ValueError):
        predictor.predict(_random_image(), topk=99)  # > num_classes


def test_predictor_fixed_resolution_model(tmp_path):
    ckpt = tmp_path / "best.pt"
    model = create_model("efficientvit_m0", num_classes=5, image_size=64)
    save_model(ckpt, model)
    predictor = Predictor.from_checkpoint(
        ckpt, model_name="efficientvit_m0", num_classes=5, image_size=64, device="cpu",
    )
    probs, indices = predictor.predict(_random_image(120), topk=2)
    assert probs.shape == (2,)
