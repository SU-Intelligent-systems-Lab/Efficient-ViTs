"""Tests for multi-scale sampling and the dynamic-input gating it relies on."""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data import MultiScaleSampler, build_multi_scale_loader, make_train_transform_builder
from models import supports_dynamic_input


class _FakePILDataset(Dataset):
    """Tiny dataset returning (PIL image, label), like an ImageFolder."""

    def __init__(self, n=40):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        arr = (np.random.rand(40, 40, 3) * 255).astype("uint8")
        return Image.fromarray(arr), index % 3


def test_batch_size_scaling_formula():
    # b(s) = max(1, round(b0 * s0^2 / s^2)).
    sampler = MultiScaleSampler(
        dataset_length=100, scales=(64, 128), base_size=128, base_batch_size=8
    )
    # At base size -> base batch; at half size -> 4x batch (area scales as s^2).
    assert sampler.batch_size_per_scale[128] == 8
    assert sampler.batch_size_per_scale[64] == 32


def test_sampler_is_deterministic_per_epoch():
    kwargs = dict(dataset_length=100, scales=(32, 64), base_size=64, base_batch_size=4, seed=123)
    a = MultiScaleSampler(**kwargs)
    b = MultiScaleSampler(**kwargs)
    assert list(a) == list(b)

    # Re-deriving the same epoch gives the same schedule; a different epoch differs.
    schedule0 = list(a)
    a.set_epoch(1)
    schedule1 = list(a)
    a.set_epoch(0)
    assert list(a) == schedule0
    assert schedule1 != schedule0


def test_sampler_batches_use_scale_specific_size():
    sampler = MultiScaleSampler(
        dataset_length=64, scales=(32, 64), base_size=64, base_batch_size=4
    )
    for batch in sampler:
        # Each entry is (height, width, dataset_index); all share one scale.
        heights = {h for (h, w, _) in batch}
        widths = {w for (h, w, _) in batch}
        assert len(heights) == 1 and len(widths) == 1
        scale = next(iter(heights))
        assert scale in (32, 64)
        # The batch length matches the scale's configured batch size.
        assert len(batch) == sampler.batch_size_per_scale[scale]


def test_multi_scale_loader_yields_valid_sizes():
    dataset = _FakePILDataset(n=40)
    loader = build_multi_scale_loader(
        dataset,
        transform_builder=make_train_transform_builder(),
        scales=(32, 64),
        base_size=64,
        base_batch_size=4,
        num_workers=0,
        seed=0,
    )
    assert len(loader) > 0
    for images, labels in loader:
        # Images are square at one of the configured scales.
        assert images.ndim == 4
        assert images.shape[-1] in (32, 64)
        assert images.shape[-1] == images.shape[-2]
        assert images.shape[0] == labels.shape[0]


def test_dynamic_input_gating_for_mixed_size():
    # Only dynamic-input models may take part in multi-scale training.
    assert supports_dynamic_input("mobilevit_s") is True
    assert supports_dynamic_input("torchvision_efficientnet_b0") is True
    assert supports_dynamic_input("efficientvit_m0") is False
    assert supports_dynamic_input("efficientformer_l1") is False
