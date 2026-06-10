"""
This module provides:
- Predictor: load any registered model and run inference on one or more images.

The Predictor wraps a model (in eval mode) together with the deterministic
evaluation transform. Each call applies the transform, runs a forward pass under
``torch.no_grad``, and returns top-k softmax probabilities and class indices. It
works with **every** model in the library because it builds models through the
unified :func:`models.registry.create_model` and relies only on the unified
contract.

Single-image vs. batched inference
----------------------------------
- ``predict(image, topk)`` runs one image. Returns two tensors of shape
  ``(topk,)``: probabilities and class indices.
- ``predict_batch(images, topk)`` runs a list of images in one forward pass.
  Returns two tensors of shape ``(B, topk)``.

Both apply the same transform pipeline; the batched form stacks the tensors
before the forward pass.

Loading a trained model
-----------------------
``Predictor.from_checkpoint`` is the usual entry point: it builds the named
model, loads a ``best.pt`` weights file (the only checkpoint this library
writes), and returns a ready-to-use predictor.

    predictor = Predictor.from_checkpoint(
        "best.pt", model_name="efficientvit_m0", num_classes=100, image_size=224,
    )

Image size and fixed-resolution models
---------------------------------------
For fixed-resolution models (EfficientViT, EfficientFormer) the ``image_size``
used to build the model and the size produced by the eval transform must match;
``from_checkpoint`` keeps them consistent automatically.

Example
-------
>>> from PIL import Image
>>> from inference.predictor import Predictor
>>>
>>> predictor = Predictor.from_checkpoint(
...     "best.pt", model_name="mobilevit_xs", num_classes=100, image_size=256,
... )
>>> image = Image.open("sample.jpg").convert("RGB")
>>> probs, indices = predictor.predict(image, topk=5)
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from PIL.Image import Image as PILImage
from torch import Tensor, nn

from data.transforms import make_eval_transform
from models.pretrained import PRETRAINED_IMAGE_SIZE
from models.registry import create_model, get_model_info
from training.checkpoint import load_model


__all__ = ["Predictor"]


class Predictor:
    """
    Run inference with any registered model.

    Parameters
    ----------
    model:
        Trained model (any model satisfying the unified contract). The Predictor
        moves it to ``device`` and switches it to eval mode.

    image_size:
        Spatial size used by the evaluation transform.

    device:
        Inference device. If None, CUDA when available else CPU.

    class_names:
        Optional sequence of class names. When provided, :meth:`class_name`
        maps an integer index to a human-readable label.

    mean, std:
        Optional normalization overrides. When None, the ImageNet defaults are
        used (matching how the models are trained in this library).
    """

    def __init__(
        self,
        model: nn.Module,
        image_size: int = 224,
        device: str | torch.device | None = None,
        class_names: Sequence[str] | None = None,
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be a positive integer.")

        self.device = (
            torch.device(device) if device is not None else self._default_device()
        )
        self.model = model.to(self.device).eval()
        self.image_size = image_size
        # Only pass overrides that were provided so the ImageNet defaults stand.
        transform_kwargs: dict[str, tuple[float, float, float]] = {}
        if mean is not None:
            transform_kwargs["mean"] = mean
        if std is not None:
            transform_kwargs["std"] = std
        self.transform = make_eval_transform(image_size, **transform_kwargs)
        self.class_names: tuple[str, ...] | None = (
            tuple(class_names) if class_names is not None else None
        )

    @staticmethod
    def _default_device() -> torch.device:
        """Return CUDA when available, otherwise CPU."""
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    @staticmethod
    def _logits(output) -> Tensor:
        """Reduce a possibly-tuple model output to a single logits tensor."""
        if torch.is_tensor(output):
            return output
        return sum(output) / len(output)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        model_name: str,
        num_classes: int,
        image_size: int | None = None,
        in_channels: int = 3,
        device: str | torch.device | None = None,
        class_names: Sequence[str] | None = None,
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
        **model_kwargs,
    ) -> "Predictor":
        """
        Build a model by name, load weights from ``path``, and wrap in a Predictor.

        Parameters
        ----------
        path:
            Path to the saved model weights (a ``best.pt`` state_dict).

        model_name:
            Registry model name to construct (must match the saved architecture).

        num_classes:
            Number of classes the checkpoint was trained with.

        image_size:
            Input resolution. For fixed-resolution models this both builds the
            model and sizes the transform. When None, the model's registered
            default size is used.

        in_channels:
            Number of input channels. Default 3.

        device, class_names, mean, std:
            Forwarded to the constructor.

        **model_kwargs:
            Extra keyword arguments forwarded to ``create_model``.
        """
        # Resolve a concrete size: required for fixed-resolution models, and
        # also used to size the eval transform.
        size = image_size if image_size is not None else get_model_info(model_name).default_image_size

        model = create_model(
            model_name,
            num_classes=num_classes,
            in_channels=in_channels,
            image_size=size,
            **model_kwargs,
        )
        load_model(path, model, device=device)

        return cls(
            model=model,
            image_size=size,
            device=device,
            class_names=class_names,
            mean=mean,
            std=std,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        num_classes: int | None = None,
        image_size: int | None = None,
        in_channels: int = 3,
        device: str | torch.device | None = None,
        class_names: Sequence[str] | None = None,
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
        **model_kwargs,
    ) -> "Predictor":
        """
        Build a Predictor from a model's *bundled* local weights.

        Convenience wrapper around ``create_model(model_name, pretrained=True)``:
        it loads the ImageNet-100 checkpoint shipped in ``weights/<name>.pt`` for
        the hand-built models (or the torchvision ImageNet-1k weights), with no
        explicit checkpoint path needed.

        Parameters
        ----------
        model_name:
            Registry model name with bundled weights (see
            ``models.list_pretrained()``).

        num_classes:
            Number of classes. When None, the checkpoint's own class count is
            used (the trained head is kept).

        image_size:
            Inference size. When None, the weights' training resolution (224) is
            used.

        in_channels, device, class_names, mean, std:
            Forwarded to the constructor / model builder.

        **model_kwargs:
            Extra keyword arguments forwarded to ``create_model``.
        """
        model = create_model(
            model_name,
            num_classes=num_classes,
            in_channels=in_channels,
            image_size=image_size,
            pretrained=True,
            **model_kwargs,
        )
        size = image_size if image_size is not None else PRETRAINED_IMAGE_SIZE
        return cls(
            model=model,
            image_size=size,
            device=device,
            class_names=class_names,
            mean=mean,
            std=std,
        )

    @torch.no_grad()
    def predict(self, image: PILImage, topk: int = 1) -> tuple[Tensor, Tensor]:
        """
        Run inference on a single image.

        Pipeline:
        1. Eval transform: PIL -> tensor (3, size, size).
        2. Add a batch dim: -> (1, 3, size, size).
        3. Forward: -> logits (1, num_classes).
        4. Softmax over classes and squeeze: -> (num_classes,).
        5. Top-k selection.

        Parameters
        ----------
        image:
            PIL image in RGB mode.

        topk:
            Number of top predictions to return.

        Returns
        -------
        probs:
            Tensor (topk,) of softmax probabilities, highest first.

        indices:
            Tensor (topk,) of class indices, sorted by probability.
        """
        if topk <= 0:
            raise ValueError("topk must be a positive integer.")

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self._logits(self.model(tensor))
        probs = logits.softmax(dim=1).squeeze(0)

        if topk > probs.shape[0]:
            raise ValueError("topk cannot be larger than the number of classes.")

        top_probs, top_indices = probs.topk(topk, largest=True, sorted=True)
        return top_probs, top_indices

    @torch.no_grad()
    def predict_batch(
        self,
        images: Sequence[PILImage],
        topk: int = 1,
    ) -> tuple[Tensor, Tensor]:
        """
        Run inference on a batch of images in one forward pass.

        Parameters
        ----------
        images:
            Sequence of PIL images in RGB mode.

        topk:
            Number of top predictions to return per image.

        Returns
        -------
        probs:
            Tensor (B, topk) of softmax probabilities.

        indices:
            Tensor (B, topk) of class indices.
        """
        if topk <= 0:
            raise ValueError("topk must be a positive integer.")

        if len(images) == 0:
            raise ValueError("images must contain at least one image.")

        # Transform each image and stack into one (B, 3, size, size) batch.
        tensor = torch.stack([self.transform(img) for img in images]).to(self.device)
        logits = self._logits(self.model(tensor))
        probs = logits.softmax(dim=1)

        if topk > probs.shape[1]:
            raise ValueError("topk cannot be larger than the number of classes.")

        top_probs, top_indices = probs.topk(topk, dim=1, largest=True, sorted=True)
        return top_probs, top_indices

    def class_name(self, index: int | Tensor) -> str:
        """
        Convert a class index to a class name.

        If no class names were provided, returns the string representation of
        the integer index instead.
        """
        if isinstance(index, Tensor):
            index = int(index.item())
        else:
            index = int(index)

        if self.class_names is None:
            return str(index)

        if not 0 <= index < len(self.class_names):
            raise IndexError(
                f"class index {index} is out of range for "
                f"{len(self.class_names)} classes."
            )

        return self.class_names[index]
