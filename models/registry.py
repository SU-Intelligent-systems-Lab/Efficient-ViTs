"""
Unified model registry: one entry point to build every model in the library.

Instead of importing a specific factory, callers ask the registry for a model
by *name*:

    from models.registry import create_model, list_models
    model = create_model("efficientvit_m0", num_classes=100, image_size=224)

This is what lets the trainer, predictor, latency benchmark, and tests treat
all four model families uniformly. Each registered model records two pieces of
metadata that the rest of the library relies on:

- ``dynamic_input``: whether the model accepts arbitrary spatial sizes. MobileViT
  and the torchvision CNNs do (they use adaptive pooling / dynamic patch
  folding); EfficientViT and EfficientFormer do **not** (their attention biases
  are tied to a fixed resolution). The multi-scale sampler queries this flag via
  :func:`supports_dynamic_input` and refuses fixed-resolution models.
- ``default_image_size``: a sensible training/benchmark resolution.

Naming convention
-----------------
- MobileViT:        ``mobilevit_xxs``, ``mobilevit_xs``, ``mobilevit_s``
- EfficientViT:     ``efficientvit_m0`` .. ``efficientvit_m5``
- EfficientFormer:  ``efficientformer_l1``, ``_l3``, ``_l7``, ``_l3_mini``, ``_l7_mini``
- torchvision:      ``torchvision_<arch>`` (e.g. ``torchvision_efficientnet_b0``)

Example
-------
>>> from models.registry import create_model, list_models, supports_dynamic_input
>>> import torch
>>>
>>> model = create_model("mobilevit_xs", num_classes=10)
>>> model(torch.randn(1, 3, 64, 64)).shape
torch.Size([1, 10])
>>> "efficientvit_m0" in list_models(family="efficientvit")
True
>>> supports_dynamic_input("efficientvit_m0")
False
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn

from . import configs
from .efficientformer import (
    EfficientFormer,
    efficientformer_l1,
    efficientformer_l3,
    efficientformer_l3_mini,
    efficientformer_l7,
    efficientformer_l7_mini,
)
from .efficientvit import EfficientViT
from .mobilevit import mobilevit_s, mobilevit_xs, mobilevit_xxs
from .pretrained import (
    PRETRAINED_IMAGE_SIZE,
    get_pretrained_path,
    has_pretrained,
    infer_num_classes,
    list_pretrained,
)
from .torchvision_models import TORCHVISION_MODELS, build_torchvision_model


__all__ = [
    "create_model",
    "list_models",
    "list_families",
    "supports_dynamic_input",
    "get_model_info",
    "ModelInfo",
    "list_pretrained",
    "has_pretrained",
]


# A builder takes the unified keyword arguments and returns an nn.Module.
ModelBuilder = Callable[..., nn.Module]


@dataclass(frozen=True)
class ModelInfo:
    """
    Registry metadata for one model name.

    Attributes
    ----------
    builder:
        Callable ``(num_classes, in_channels, image_size, pretrained, **kwargs)
        -> nn.Module``.

    family:
        Coarse family label: ``"mobilevit"``, ``"efficientvit"``,
        ``"efficientformer"`` or ``"torchvision"``.

    dynamic_input:
        True if the model accepts arbitrary input spatial sizes. False for
        fixed-resolution models (EfficientViT, EfficientFormer).

    default_image_size:
        Recommended training/benchmark resolution.

    pretrained_available:
        Whether ``pretrained=True`` is supported. True for torchvision models
        (ImageNet-1k weights) and for any hand-built model that has a local
        weights file in ``weights/<name>.pt`` (the ImageNet-100 checkpoints
        trained with this library).
    """

    builder: ModelBuilder
    family: str
    dynamic_input: bool
    default_image_size: int = 224
    pretrained_available: bool = False


# ---------------------------------------------------------------------------
# Builder factories. Each returns a closure with the unified signature so that
# create_model can call every model the same way regardless of family.
# ---------------------------------------------------------------------------


def _mobilevit_builder(factory: Callable[..., nn.Module]) -> ModelBuilder:
    """Wrap a MobileViT factory. MobileViT ignores image_size (it is dynamic)."""

    def build(
        num_classes: int,
        in_channels: int,
        image_size: int | None,
        pretrained: bool,
        **kwargs: Any,
    ) -> nn.Module:
        # ``pretrained`` is handled upstream by create_model (it loads local
        # weights); the builder always constructs a fresh model. image_size is
        # unused here -- MobileViT handles any resolution.
        return factory(num_classes=num_classes, in_channels=in_channels, **kwargs)

    return build


def _efficientvit_builder(config_fn: Callable[[], "configs.EfficientViTConfig"]) -> ModelBuilder:
    """Wrap an EfficientViT config factory, overriding img_size when requested."""

    def build(
        num_classes: int,
        in_channels: int,
        image_size: int | None,
        pretrained: bool,
        **kwargs: Any,
    ) -> nn.Module:
        # ``pretrained`` is handled upstream by create_model (local-weights load).
        config = config_fn()
        # EfficientViT bakes the resolution into its config; honour an explicit
        # image_size by rebuilding the (frozen) config with the new img_size.
        if image_size is not None and image_size != config.img_size:
            config = replace(config, img_size=image_size)
        return EfficientViT(config, num_classes=num_classes, in_channels=in_channels, **kwargs)

    return build


def _efficientformer_builder(factory: Callable[..., nn.Module]) -> ModelBuilder:
    """Wrap an EfficientFormer factory, passing img_size through."""

    def build(
        num_classes: int,
        in_channels: int,
        image_size: int | None,
        pretrained: bool,
        **kwargs: Any,
    ) -> nn.Module:
        # ``pretrained`` is handled upstream by create_model (local-weights load).
        img_size = image_size if image_size is not None else 224
        return factory(num_classes=num_classes, img_size=img_size, in_channels=in_channels, **kwargs)

    return build


def _torchvision_builder(name: str) -> ModelBuilder:
    """Wrap a torchvision model name. torchvision models are dynamic-input."""

    def build(
        num_classes: int,
        in_channels: int,
        image_size: int | None,
        pretrained: bool,
        **kwargs: Any,
    ) -> nn.Module:
        if kwargs:
            raise TypeError(
                f"torchvision model {name!r} does not accept extra kwargs {list(kwargs)}."
            )
        # image_size is intentionally unused: adaptive pooling handles any size.
        return build_torchvision_model(
            name, num_classes=num_classes, in_channels=in_channels, pretrained=pretrained
        )

    return build


def _build_registry() -> dict[str, ModelInfo]:
    """Assemble the full {name: ModelInfo} table for all four families."""
    registry: dict[str, ModelInfo] = {}

    # MobileViT (dynamic input).
    for name, factory in (
        ("mobilevit_xxs", mobilevit_xxs),
        ("mobilevit_xs", mobilevit_xs),
        ("mobilevit_s", mobilevit_s),
    ):
        registry[name] = ModelInfo(
            builder=_mobilevit_builder(factory),
            family="mobilevit",
            dynamic_input=True,
            default_image_size=256,
            pretrained_available=has_pretrained(name),
        )

    # EfficientViT (fixed input).
    for name in ("m0", "m1", "m2", "m3", "m4", "m5"):
        config_fn = getattr(configs, f"efficientvit_{name}_config")
        registry[f"efficientvit_{name}"] = ModelInfo(
            builder=_efficientvit_builder(config_fn),
            family="efficientvit",
            dynamic_input=False,
            default_image_size=224,
            pretrained_available=has_pretrained(f"efficientvit_{name}"),
        )

    # EfficientFormer (fixed input).
    for name, factory in (
        ("efficientformer_l1", efficientformer_l1),
        ("efficientformer_l3", efficientformer_l3),
        ("efficientformer_l7", efficientformer_l7),
        ("efficientformer_l3_mini", efficientformer_l3_mini),
        ("efficientformer_l7_mini", efficientformer_l7_mini),
    ):
        registry[name] = ModelInfo(
            builder=_efficientformer_builder(factory),
            family="efficientformer",
            dynamic_input=False,
            default_image_size=224,
            pretrained_available=has_pretrained(name),
        )

    # torchvision CNNs (dynamic input, pretrained weights available).
    for name, spec in TORCHVISION_MODELS.items():
        registry[name] = ModelInfo(
            builder=_torchvision_builder(name),
            family="torchvision",
            dynamic_input=True,
            default_image_size=spec.default_image_size,
            pretrained_available=True,
        )

    return registry


# The single source of truth for "what models exist".
_REGISTRY: dict[str, ModelInfo] = _build_registry()


def create_model(
    name: str,
    num_classes: int | None = None,
    in_channels: int = 3,
    image_size: int | None = None,
    pretrained: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """
    Build a model by registry name.

    Parameters
    ----------
    name:
        Registered model name (see :func:`list_models`).

    num_classes:
        Number of output classes for the classification head. When ``None``
        (the default), it resolves to the checkpoint's class count for a
        ``pretrained=True`` hand-built model (so the trained head is kept), and
        to ``1000`` otherwise.

    in_channels:
        Number of input image channels. Hand-written models support any value;
        torchvision builders support only 3.

    image_size:
        Input resolution. Required-but-defaulted for fixed-resolution models
        (EfficientViT/EfficientFormer), where it determines the architecture;
        ignored by dynamic-input models. When None, the model's
        ``default_image_size`` is used for fixed-resolution models. For a
        ``pretrained=True`` hand-built model it defaults to the resolution the
        bundled weights were trained at (224).

    pretrained:
        Load pretrained weights:
        - **torchvision** models load ImageNet-1k weights from torchvision.
        - **MobileViT / EfficientViT / EfficientFormer** load the local
          ImageNet-100 checkpoint bundled in ``weights/<name>.pt`` (the models
          trained with this library). If no local file exists for that name a
          ``FileNotFoundError`` is raised listing what *is* available.

    **kwargs:
        Extra keyword arguments forwarded to the underlying model constructor
        (e.g. ``drop_path`` for MobileViT, ``distillation`` for EfficientViT).

    Returns
    -------
    nn.Module
        The constructed model, satisfying the unified contract
        (``forward``/``forward_features``/``reset_classifier``/``num_features``).
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown model {name!r}. Use list_models() to see the "
            f"{len(_REGISTRY)} available models."
        )
    info = _REGISTRY[name]

    # Hand-built families load their pretrained weights from local files; this
    # is handled here (not in the builder) because the logic is identical across
    # the three families.
    if pretrained and info.family != "torchvision":
        return _load_local_pretrained(name, info, num_classes, in_channels, image_size, **kwargs)

    # Default class count for the non-local-pretrained path.
    resolved_classes = num_classes if num_classes is not None else 1000

    # For fixed-resolution models, fall back to the recommended size so that
    # `create_model("efficientvit_m0")` works without a manual image_size.
    if image_size is None and not info.dynamic_input:
        image_size = info.default_image_size

    return info.builder(
        num_classes=resolved_classes,
        in_channels=in_channels,
        image_size=image_size,
        pretrained=pretrained,
        **kwargs,
    )


def _load_local_pretrained(
    name: str,
    info: ModelInfo,
    num_classes: int | None,
    in_channels: int,
    image_size: int | None,
    **kwargs: Any,
) -> nn.Module:
    """
    Build a hand-built model and load its local ImageNet-100 weights.

    The model is first built at the checkpoint's *native* class count and at the
    resolution the weights were trained at (224 for fixed-resolution models), the
    weights are loaded strictly, and finally — if the caller asked for a
    different ``num_classes`` — a fresh classifier head is swapped in via
    ``reset_classifier``.

    Raises
    ------
    FileNotFoundError
        If there are no local weights for ``name``.
    """
    path = get_pretrained_path(name)
    if path is None:
        raise FileNotFoundError(
            f"No local pretrained weights for {name!r} (looked in "
            f"weights/{name}.pt). Models with bundled weights: {list_pretrained()}."
        )

    # Read the checkpoint and recover the class count it was trained with.
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint_classes = infer_num_classes(state_dict)

    # Fixed-resolution models must be rebuilt at the trained resolution so their
    # attention-bias tables match; dynamic models are unaffected by the choice.
    build_size = image_size if image_size is not None else PRETRAINED_IMAGE_SIZE

    model = info.builder(
        num_classes=checkpoint_classes,
        in_channels=in_channels,
        image_size=build_size,
        pretrained=False,
        **kwargs,
    )
    model.load_state_dict(state_dict, strict=True)

    # Retarget the head only if the caller explicitly asked for a different count.
    target_classes = num_classes if num_classes is not None else checkpoint_classes
    if target_classes != checkpoint_classes:
        model.reset_classifier(target_classes)

    return model


def list_models(
    family: str | None = None,
    dynamic_input: bool | None = None,
) -> list[str]:
    """
    List registered model names, optionally filtered.

    Parameters
    ----------
    family:
        If given, return only models in this family ("mobilevit",
        "efficientvit", "efficientformer", "torchvision").

    dynamic_input:
        If given, return only models whose ``dynamic_input`` flag matches. Pass
        ``True`` to list models safe for multi-scale sampling.

    Returns
    -------
    list[str]
        Sorted list of model names matching the filters.
    """
    names = []
    for name, info in _REGISTRY.items():
        if family is not None and info.family != family:
            continue
        if dynamic_input is not None and info.dynamic_input != dynamic_input:
            continue
        names.append(name)
    return sorted(names)


def list_families() -> list[str]:
    """Return the sorted list of distinct model families."""
    return sorted({info.family for info in _REGISTRY.values()})


def supports_dynamic_input(name: str) -> bool:
    """
    Return whether ``name`` accepts arbitrary input spatial sizes.

    Used by the multi-scale data loader to decide whether a model may take part
    in variable-resolution training. Fixed-resolution models (EfficientViT,
    EfficientFormer) return False.
    """
    return get_model_info(name).dynamic_input


def get_model_info(name: str) -> ModelInfo:
    """Return the :class:`ModelInfo` metadata for ``name``."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Use list_models() to see options.")
    return _REGISTRY[name]
