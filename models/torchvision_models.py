"""
Clean builders for torchvision classification models, wrapped to satisfy the
library's unified model contract.

torchvision ships excellent, well-tested CNN baselines. Rather than
re-implementing them, this module wraps each one in :class:`TorchvisionModel`,
a thin adapter that gives every torchvision model the same interface as the
hand-written models in this library:

- ``forward(x)``                    -> logits ``(B, num_classes)``
- ``forward_features(x)``           -> pooled feature vector ``(B, num_features)``
- ``reset_classifier(num_classes)`` -> swap in a fresh linear head
- ``num_features``                  -> width of the pooled feature vector

This lets MobileViT, EfficientViT, EfficientFormer, and torchvision CNNs all be
created, trained, benchmarked, and used for inference through exactly the same
code paths.

Why these models are dynamic-input-safe
---------------------------------------
Every torchvision model here ends its feature extractor with an
``AdaptiveAvgPool2d``, which collapses any spatial size to ``1x1`` before the
classifier. They therefore accept arbitrary input resolutions and are safe to
use with the multi-scale sampler -- unlike EfficientViT/EfficientFormer.

How ``forward_features`` works
------------------------------
A torchvision model's ``forward`` runs ``backbone -> pool -> head`` internally
with heterogeneous head attribute names (``fc`` for ShuffleNet, ``classifier``
for the rest). To extract the pre-head pooled vector without re-implementing
each forward, the adapter temporarily swaps the head for ``nn.Identity`` and
runs the model's own ``forward`` (restoring the head afterwards in a
``finally``). The result is the exact tensor that would have entered the head.

Available models
----------------
- ShuffleNetV2: x0.5, x1.0, x1.5, x2.0
- EfficientNet: b0..b7, v2_s, v2_m, v2_l
- MobileNetV2
- MobileNetV3: small, large

Example
-------
>>> import torch
>>> from models.torchvision_models import build_torchvision_model, list_torchvision_models
>>>
>>> "torchvision_mobilenet_v3_large" in list_torchvision_models()
True
>>> model = build_torchvision_model("torchvision_efficientnet_b0", num_classes=100)
>>> model(torch.randn(2, 3, 96, 96)).shape       # adaptive pool -> any size works
torch.Size([2, 100])
>>> model.forward_features(torch.randn(2, 3, 96, 96)).shape
torch.Size([2, 1280])
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch import Tensor, nn
from torchvision import models as tv_models

from .common import ensure_positive_int


__all__ = [
    "TorchvisionModel",
    "build_torchvision_model",
    "list_torchvision_models",
    "TORCHVISION_MODELS",
]


@dataclass(frozen=True)
class _TorchvisionSpec:
    """
    Static description of how to build and adapt one torchvision model.

    Parameters
    ----------
    factory:
        The torchvision constructor (e.g. ``torchvision.models.efficientnet_b0``).

    head_attr:
        Name of the attribute holding the classification head on the
        torchvision model -- ``"fc"`` for ShuffleNet, ``"classifier"`` for the
        EfficientNet / MobileNet families.

    default_image_size:
        The resolution the model was designed/trained for, used only for
        documentation and benchmark defaults (the models accept any size).
    """

    factory: Callable[..., nn.Module]
    head_attr: str
    default_image_size: int = 224


def _find_in_features(head: nn.Module) -> int:
    """
    Return the input feature count of a torchvision classification head.

    Handles the two head shapes used by torchvision:
    - a bare ``nn.Linear`` (ShuffleNet's ``fc``): use its ``in_features``.
    - an ``nn.Sequential`` (MobileNet / EfficientNet ``classifier``): use the
      ``in_features`` of its first ``nn.Linear`` submodule.
    """
    if isinstance(head, nn.Linear):
        return head.in_features
    if isinstance(head, nn.Sequential):
        for module in head:
            if isinstance(module, nn.Linear):
                return module.in_features
    raise TypeError(
        f"Could not determine in_features for head of type {type(head).__name__}."
    )


class TorchvisionModel(nn.Module):
    """
    Adapter that gives a torchvision classification model the unified contract.

    Parameters
    ----------
    model:
        An instantiated torchvision classification model.

    head_attr:
        Attribute name of the classification head on ``model`` (``"fc"`` or
        ``"classifier"``).

    Attributes
    ----------
    num_features:
        Width of the pooled feature vector entering the classification head.

    num_classes:
        Current number of output classes.
    """

    def __init__(self, model: nn.Module, head_attr: str) -> None:
        super().__init__()
        self.model = model
        self._head_attr = head_attr

        head = getattr(model, head_attr)
        # Pre-head feature width and the current class count.
        self.num_features = _find_in_features(head)
        self.num_classes = self._infer_num_classes(head)

    @staticmethod
    def _infer_num_classes(head: nn.Module) -> int:
        """Read the output dimension (num_classes) from a head module."""
        if isinstance(head, nn.Linear):
            return head.out_features
        if isinstance(head, nn.Sequential):
            # The last Linear in the head produces the logits.
            for module in reversed(list(head)):
                if isinstance(module, nn.Linear):
                    return module.out_features
        raise TypeError("Could not determine num_classes from the head module.")

    def reset_classifier(self, num_classes: int) -> None:
        """
        Replace the entire classification head with a single ``nn.Linear``.

        This unifies head behaviour across architectures: even MobileNetV3,
        whose native head has an extra hidden layer, is reduced to one
        ``Linear(num_features, num_classes)``. Pass ``num_classes <= 0`` to drop
        the head (``nn.Identity``), making ``forward`` return pooled features.

        Parameters
        ----------
        num_classes:
            New number of output classes.
        """
        self.num_classes = num_classes
        new_head: nn.Module = (
            nn.Identity() if num_classes <= 0 else nn.Linear(self.num_features, num_classes)
        )
        setattr(self.model, self._head_attr, new_head)

    def forward_features(self, x: Tensor) -> Tensor:
        """
        Return the pooled feature vector that feeds the classification head.

        Implemented by temporarily swapping the head for ``nn.Identity`` and
        running the model's own ``forward`` (the head is always restored, even
        if the forward pass raises).

        Input shape:  (B, in_channels, H, W)   (H, W arbitrary; adaptive pool)
        Output shape: (B, num_features)
        """
        original_head = getattr(self.model, self._head_attr)
        setattr(self.model, self._head_attr, nn.Identity())
        try:
            return self.model(x)
        finally:
            setattr(self.model, self._head_attr, original_head)

    def forward(self, x: Tensor) -> Tensor:
        """
        Run the wrapped torchvision model.

        Input shape:  (B, in_channels, H, W)
        Output shape: (B, num_classes)
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected input shape (B, C, H, W), got {tuple(x.shape)}."
            )
        return self.model(x)


# Static table of every supported torchvision model, keyed by the name used in
# the unified registry. The "torchvision_" prefix keeps these names distinct
# from the hand-written models.
TORCHVISION_MODELS: dict[str, _TorchvisionSpec] = {
    # ShuffleNetV2 -- head attribute is "fc".
    "torchvision_shufflenet_v2_x0_5": _TorchvisionSpec(tv_models.shufflenet_v2_x0_5, "fc"),
    "torchvision_shufflenet_v2_x1_0": _TorchvisionSpec(tv_models.shufflenet_v2_x1_0, "fc"),
    "torchvision_shufflenet_v2_x1_5": _TorchvisionSpec(tv_models.shufflenet_v2_x1_5, "fc"),
    "torchvision_shufflenet_v2_x2_0": _TorchvisionSpec(tv_models.shufflenet_v2_x2_0, "fc"),
    # EfficientNet B0..B7 -- head attribute is "classifier".
    "torchvision_efficientnet_b0": _TorchvisionSpec(tv_models.efficientnet_b0, "classifier"),
    "torchvision_efficientnet_b1": _TorchvisionSpec(tv_models.efficientnet_b1, "classifier"),
    "torchvision_efficientnet_b2": _TorchvisionSpec(tv_models.efficientnet_b2, "classifier"),
    "torchvision_efficientnet_b3": _TorchvisionSpec(tv_models.efficientnet_b3, "classifier"),
    "torchvision_efficientnet_b4": _TorchvisionSpec(tv_models.efficientnet_b4, "classifier"),
    "torchvision_efficientnet_b5": _TorchvisionSpec(tv_models.efficientnet_b5, "classifier"),
    "torchvision_efficientnet_b6": _TorchvisionSpec(tv_models.efficientnet_b6, "classifier"),
    "torchvision_efficientnet_b7": _TorchvisionSpec(tv_models.efficientnet_b7, "classifier"),
    # EfficientNetV2 -- head attribute is "classifier".
    "torchvision_efficientnet_v2_s": _TorchvisionSpec(tv_models.efficientnet_v2_s, "classifier"),
    "torchvision_efficientnet_v2_m": _TorchvisionSpec(tv_models.efficientnet_v2_m, "classifier"),
    "torchvision_efficientnet_v2_l": _TorchvisionSpec(tv_models.efficientnet_v2_l, "classifier"),
    # MobileNetV2 / V3 -- head attribute is "classifier".
    "torchvision_mobilenet_v2": _TorchvisionSpec(tv_models.mobilenet_v2, "classifier"),
    "torchvision_mobilenet_v3_small": _TorchvisionSpec(tv_models.mobilenet_v3_small, "classifier"),
    "torchvision_mobilenet_v3_large": _TorchvisionSpec(tv_models.mobilenet_v3_large, "classifier"),
}


def list_torchvision_models() -> list[str]:
    """Return the sorted list of registered torchvision model names."""
    return sorted(TORCHVISION_MODELS)


def build_torchvision_model(
    name: str,
    num_classes: int = 1000,
    in_channels: int = 3,
    pretrained: bool = False,
) -> TorchvisionModel:
    """
    Build a torchvision model wrapped in :class:`TorchvisionModel`.

    Parameters
    ----------
    name:
        A key of :data:`TORCHVISION_MODELS`, e.g. ``"torchvision_efficientnet_b0"``.

    num_classes:
        Number of output classes for the head.

    in_channels:
        Number of input channels. Only 3 is supported: torchvision stems are
        built for RGB and swapping the first conv per-architecture is out of
        scope. A clear error is raised for any other value.

    pretrained:
        If True, load ImageNet-1k pretrained weights (downloaded by torchvision)
        and then, if ``num_classes != 1000``, replace the head via
        ``reset_classifier``. If False, the model is randomly initialised.

    Returns
    -------
    TorchvisionModel
        The wrapped model satisfying the unified contract.
    """
    ensure_positive_int(num_classes, "num_classes")
    ensure_positive_int(in_channels, "in_channels")
    if name not in TORCHVISION_MODELS:
        raise ValueError(
            f"Unknown torchvision model {name!r}. Available: {list_torchvision_models()}"
        )
    if in_channels != 3:
        raise ValueError(
            "torchvision builders support only in_channels=3 (RGB); "
            f"got {in_channels}."
        )

    spec = TORCHVISION_MODELS[name]

    if pretrained:
        # Load the default ImageNet-1k weights (1000-class head), then retarget.
        base = spec.factory(weights="DEFAULT")
        wrapper = TorchvisionModel(base, spec.head_attr)
        if num_classes != wrapper.num_classes:
            wrapper.reset_classifier(num_classes)
    else:
        # Build directly with the requested head size, random init.
        base = spec.factory(weights=None, num_classes=num_classes)
        wrapper = TorchvisionModel(base, spec.head_attr)

    return wrapper
