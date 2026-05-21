"""
This module provides:
- Predictor: load a trained MobileViT model and run inference on one or
  more images at a single fixed spatial resolution.

The Predictor wraps a model in eval mode together with the deterministic
evaluation transform pipeline. Each call applies the transform, runs a
forward pass under torch.no_grad, and returns top-k softmax probabilities
and class indices.

Example
-------
>>> from PIL import Image
>>> from mobilevit.models.mobilevit import mobilevit_s
>>> from mobilevit.inference.predictor import Predictor
>>>
>>> model = mobilevit_s(num_classes=100)
>>> predictor = Predictor.from_checkpoint("mobilevit.pt", model, size=256)
>>>
>>> image = Image.open("sample.jpg").convert("RGB")
>>> probs, indices = predictor.predict(image, topk=5)
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from PIL.Image import Image as PILImage
from torch import Tensor, nn

from data.transforms import build_eval_transforms
from training.checkpoint import load_model


__all__ = ["Predictor"]


class Predictor:
    """
    Run inference with a trained MobileViT model.

    Parameters
    ----------
    model:
        Trained model. The Predictor moves it to the chosen device and sets
        it to eval mode.

    size:
        Spatial size used by the evaluation transform pipeline.

    device:
        Device used for inference. If None, CUDA is used when available,
        otherwise CPU.

    class_names:
        Optional sequence of class names. If provided, class_name converts
        an integer index to a human-readable label.
    """

    def __init__(
        self,
        model: nn.Module,
        size: int = 256,
        device: str | torch.device | None = None,
        class_names: Sequence[str] | None = None,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be a positive integer.")

        self.device = (
            torch.device(device) if device is not None else self._default_device()
        )
        self.model = model.to(self.device).eval()
        self.transform = build_eval_transforms(size)
        self.class_names: tuple[str, ...] | None = (
            tuple(class_names) if class_names is not None else None
        )

    @staticmethod
    def _default_device() -> torch.device:
        """Return CUDA when available, otherwise CPU."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        model: nn.Module,
        size: int = 256,
        device: str | torch.device | None = None,
        class_names: Sequence[str] | None = None,
    ) -> Predictor:
        """
        Build a Predictor by loading model weights from a checkpoint.

        Parameters
        ----------
        path:
            Path to the saved model weights.

        model:
            Model instance whose weights will be loaded from the checkpoint.

        size:
            Spatial size used by the evaluation transform pipeline.

        device:
            Device used for inference. If None, CUDA is used when available.

        class_names:
            Optional sequence of class names.
        """
        load_model(path, model, device=device)
        return cls(
            model=model,
            size=size,
            device=device,
            class_names=class_names,
        )

    @torch.no_grad()
    def predict(
        self,
        image: PILImage,
        topk: int = 1,
    ) -> tuple[Tensor, Tensor]:
        """
        Run inference on a single image.

        Parameters
        ----------
        image:
            PIL image in RGB mode.

        topk:
            Number of top predictions to return.

        Returns
        -------
        probs:
            Tensor with shape (topk,) of softmax probabilities, sorted from
            highest to lowest.

        indices:
            Tensor with shape (topk,) of class indices, sorted by probability.
        """
        if topk <= 0:
            raise ValueError("topk must be a positive integer.")

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
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
        Run inference on a batch of images.

        Parameters
        ----------
        images:
            Sequence of PIL images in RGB mode.

        topk:
            Number of top predictions to return per image.

        Returns
        -------
        probs:
            Tensor with shape (B, topk) of softmax probabilities.

        indices:
            Tensor with shape (B, topk) of class indices.
        """
        if topk <= 0:
            raise ValueError("topk must be a positive integer.")

        if len(images) == 0:
            raise ValueError("images must contain at least one image.")

        tensor = torch.stack([self.transform(img) for img in images]).to(self.device)
        logits = self.model(tensor)
        probs = logits.softmax(dim=1)

        if topk > probs.shape[1]:
            raise ValueError("topk cannot be larger than the number of classes.")

        top_probs, top_indices = probs.topk(topk, dim=1, largest=True, sorted=True)
        return top_probs, top_indices

    def class_name(self, index: int | Tensor) -> str:
        """
        Convert a class index to a class name.

        Returns the string representation of the integer index when no class
        names were provided at construction time.
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
