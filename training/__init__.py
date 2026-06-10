"""
Training utilities for the Light-weight ViTs library.

This package provides:
- TrainConfig: a frozen dataclass describing one training run.
- Trainer: a model-agnostic training/validation loop with AMP, gradient
  clipping, all metrics, throughput logging, and best-only checkpointing.
- build_optimizer / build_scheduler (+ SchedulerSpec): optimizer and LR
  scheduler factories.
- accuracy, MetricTracker, ClassificationMetrics, precision_recall_f1,
  confusion_matrix: metrics.
- save_model / load_model: best-weights checkpoint I/O.
"""

from .checkpoint import load_model, save_model
from .config import TrainConfig
from .metrics import (
    ClassificationMetrics,
    MetricTracker,
    accuracy,
    confusion_matrix,
    precision_recall_f1,
)
from .optimizer import SchedulerSpec, build_optimizer, build_scheduler
from .trainer import Trainer


__all__ = [
    "TrainConfig",
    "Trainer",
    "build_optimizer",
    "build_scheduler",
    "SchedulerSpec",
    "accuracy",
    "MetricTracker",
    "ClassificationMetrics",
    "confusion_matrix",
    "precision_recall_f1",
    "save_model",
    "load_model",
]
