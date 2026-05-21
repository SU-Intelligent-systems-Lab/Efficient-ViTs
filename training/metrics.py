"""
This module provides:
- accuracy: top-k classification accuracy for logits and targets
- MetricTracker: running average tracker for losses and metrics

Example
-------
>>> import torch
>>> from mobilevit.training.metrics import accuracy, MetricTracker
>>>
>>> logits = torch.randn(8, 100)
>>> targets = torch.randint(0, 100, (8,))
>>> top1, top5 = accuracy(logits, targets, topk=(1, 5))
>>>
>>> tracker = MetricTracker("loss")
>>> tracker.update(0.8, n=8)
>>> tracker.avg
0.8
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


__all__ = [
    "accuracy",
    "MetricTracker",
]


@torch.no_grad()
def accuracy(
    logits: Tensor,
    targets: Tensor,
    topk: tuple[int, ...] = (1,),
) -> list[Tensor]:
    """
    Compute top-k classification accuracy.

    Parameters
    ----------
    logits:
        Model outputs with shape (B, num_classes).

    targets:
        Ground-truth labels with shape (B,).

    topk:
        Tuple of k values. For example, (1, 5) returns top-1 and top-5 accuracy.

    Returns
    -------
    list[Tensor]
        Accuracy values in percent, one tensor for each requested k.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape (B, num_classes), got {tuple(logits.shape)}.")

    if targets.ndim != 1:
        raise ValueError(f"targets must have shape (B,), got {tuple(targets.shape)}.")

    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets must have the same batch size.")

    if len(topk) == 0:
        raise ValueError("topk must contain at least one value.")

    num_classes = logits.shape[1]
    max_k = max(topk)

    if max_k <= 0:
        raise ValueError("top-k values must be positive integers.")

    if max_k > num_classes:
        raise ValueError("top-k cannot be larger than the number of classes.")

    batch_size = targets.shape[0]

    _, predictions = logits.topk(max_k, dim=1, largest=True, sorted=True)
    predictions = predictions.t()

    correct = predictions.eq(targets.reshape(1, -1))

    results: list[Tensor] = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results.append(correct_k.mul_(100.0 / batch_size))

    return results


@dataclass
class MetricTracker:
    """
    Track the running average of a scalar metric.

    Parameters
    ----------
    name:
        Metric name, for example "loss", "top1", or "top5".

    fmt:
        Format string used by __str__.
    """

    name: str
    fmt: str = ":.4f"
    val: float = field(default=0.0, init=False)
    avg: float = field(default=0.0, init=False)
    sum: float = field(default=0.0, init=False)
    count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be a non-empty string.")

    def reset(self) -> None:
        """Reset all tracked values."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float | Tensor, n: int = 1) -> None:
        """
        Add a new metric value.

        Parameters
        ----------
        value:
            New scalar value.

        n:
            Number of samples represented by this value.
        """
        if n <= 0:
            raise ValueError("n must be a positive integer.")

        if isinstance(value, Tensor):
            value = float(value.detach().item())

        value = float(value)

        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmt = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmt.format(name=self.name, val=self.val, avg=self.avg)
