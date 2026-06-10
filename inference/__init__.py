"""
Inference utilities for the Light-weight ViTs library.

This package provides:
- Predictor: load any registered model (via the model registry) and run top-k
  single-image or batched inference, with checkpoint loading, class-name
  mapping, device selection, and image-size selection.
"""

from .predictor import Predictor


__all__ = ["Predictor"]
