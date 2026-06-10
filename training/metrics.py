"""
This module provides:
- accuracy: top-k classification accuracy for a batch of logits and targets.
- MetricTracker: running average tracker for losses and metrics.
- confusion_matrix: accumulate a (num_classes x num_classes) count matrix.
- precision_recall_f1: derive precision / recall / F1 from a confusion matrix.
- ClassificationMetrics: a stateful accumulator that computes top-1/top-5
  accuracy and precision/recall/F1 (macro / micro / weighted) over an epoch.

Top-k accuracy
--------------
For a batch of B samples with C classes:
- logits has shape (B, C).
- targets has shape (B,).

A sample is correct under top-k if its target appears among the k highest
logits. Top-1 is the strictest; top-5 is the standard ImageNet metric. The
returned accuracies are percentages in [0, 100].

Precision / recall / F1
-----------------------
These are computed from a confusion matrix ``C`` where ``C[t, p]`` counts
samples with true label ``t`` predicted as ``p``. For each class ``c``:

    TP_c  = C[c, c]
    pred_c = sum_t C[t, c]      (number predicted as c)
    true_c = sum_p C[c, p]      (number truly c)
    precision_c = TP_c / pred_c
    recall_c    = TP_c / true_c
    f1_c        = 2 * precision_c * recall_c / (precision_c + recall_c)

Averaging modes combine the per-class values:
- ``macro``   : unweighted mean over classes (every class counts equally).
- ``weighted``: mean weighted by ``true_c`` (the class support).
- ``micro``   : computed from pooled counts; for single-label classification
                micro-precision = micro-recall = micro-F1 = overall accuracy.

To keep every metric on one scale, **all values in this module are reported as
percentages in [0, 100]** (precision/recall/F1 included), so they line up with
top-1/top-5 in logs.

Example
-------
>>> import torch
>>> from training.metrics import accuracy, ClassificationMetrics
>>>
>>> logits = torch.randn(8, 100)
>>> targets = torch.randint(0, 100, (8,))
>>> top1, top5 = accuracy(logits, targets, topk=(1, 5))
>>>
>>> metrics = ClassificationMetrics(num_classes=100)
>>> metrics.update(logits, targets, loss=0.7)
>>> results = metrics.compute()      # dict with loss, top1, top5, f1_macro, ...
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor


__all__ = [
    "accuracy",
    "MetricTracker",
    "confusion_matrix",
    "precision_recall_f1",
    "ClassificationMetrics",
]


@torch.no_grad()
def accuracy(
    logits: Tensor,
    targets: Tensor,
    topk: tuple[int, ...] = (1,),
) -> list[Tensor]:
    """
    Compute top-k classification accuracy for one batch.

    Parameters
    ----------
    logits:
        Model outputs with shape (B, C).

    targets:
        Ground-truth labels with shape (B,).

    topk:
        Tuple of k values. For example, (1, 5) returns top-1 and top-5
        accuracy.

    Returns
    -------
    list[Tensor]
        Accuracy values in percent, one scalar tensor per requested k.
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

    # Per-sample top-max_k predicted class indices, sorted by score.
    # predictions shape: (B, max_k)
    _, predictions = logits.topk(max_k, dim=1, largest=True, sorted=True)

    # Transpose so each row corresponds to one rank position:
    # predictions[k, b] = the k-th best prediction for sample b.
    # Shape becomes (max_k, B).
    predictions = predictions.t()

    # Broadcast targets across the rank axis to compare in one go.
    # correct[k, b] is True if the k-th-best prediction matches target_b.
    correct = predictions.eq(targets.reshape(1, -1))

    results: list[Tensor] = []
    for k in topk:
        # A sample counts as correct under top-k if any of its top-k
        # predictions matches: sum the first k rows along the rank axis.
        correct_k = correct[:k].reshape(-1).float().sum(0)
        # Convert to percentage of the batch.
        results.append(correct_k.mul_(100.0 / batch_size))

    return results


@dataclass
class MetricTracker:
    """
    Track the running average of a scalar metric.

    Useful for accumulating per-batch losses and accuracies across an
    epoch. Each ``update(value, n)`` is treated as ``n`` independent
    samples with value ``value``, so the running average is correctly
    weighted by batch size.

    Parameters
    ----------
    name:
        Metric name, for example "loss", "top1", or "top5".

    fmt:
        Format string used by ``__str__``.

    Attributes
    ----------
    val:
        Most recent value passed to ``update``.

    avg:
        Running weighted average since the last reset.

    sum:
        Sum of value * n contributions since the last reset.

    count:
        Total number of samples (sum of n) since the last reset.
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
        """Reset all tracked values to zero."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float | Tensor, n: int = 1) -> None:
        """
        Add a new metric value weighted by ``n`` samples.

        Parameters
        ----------
        value:
            New scalar value. Tensors are detached and converted to a Python float.

        n:
            Number of samples represented by this value (typically the
            batch size).
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


@torch.no_grad()
def confusion_matrix(
    logits_or_preds: Tensor,
    targets: Tensor,
    num_classes: int,
) -> Tensor:
    """
    Accumulate a confusion matrix for one batch.

    Parameters
    ----------
    logits_or_preds:
        Either logits of shape (B, C) (argmax is taken) or hard predictions of
        shape (B,).

    targets:
        Ground-truth labels of shape (B,).

    num_classes:
        Number of classes C.

    Returns
    -------
    Tensor
        Integer matrix of shape (C, C) on the CPU, where entry ``[t, p]`` counts
        samples with true label ``t`` predicted as ``p``.
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be a positive integer.")

    # Reduce logits to hard predictions if needed.
    preds = logits_or_preds.argmax(dim=1) if logits_or_preds.ndim == 2 else logits_or_preds
    preds = preds.detach().reshape(-1).cpu()
    targets = targets.detach().reshape(-1).cpu()

    if preds.shape != targets.shape:
        raise ValueError("predictions and targets must have the same number of elements.")

    # Flatten (true, pred) pairs into a single index and count with bincount;
    # this is the fast, allocation-light way to build a confusion matrix.
    indices = targets * num_classes + preds
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes).to(torch.int64)


def precision_recall_f1(
    matrix: Tensor,
    average: str = "macro",
) -> tuple[float, float, float]:
    """
    Compute precision, recall, and F1 from a confusion matrix.

    Parameters
    ----------
    matrix:
        Confusion matrix of shape (C, C), ``matrix[t, p]`` = count of true ``t``
        predicted ``p`` (as produced by :func:`confusion_matrix`).

    average:
        One of ``"macro"``, ``"micro"``, ``"weighted"``. See the module
        docstring for the definitions.

    Returns
    -------
    tuple[float, float, float]
        ``(precision, recall, f1)`` as percentages in [0, 100].
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"confusion matrix must be square (C, C), got {tuple(matrix.shape)}.")
    if average not in {"macro", "micro", "weighted"}:
        raise ValueError(f"average must be 'macro', 'micro', or 'weighted', got {average!r}.")

    matrix = matrix.to(torch.float64)
    true_positive = torch.diag(matrix)          # TP_c
    predicted = matrix.sum(dim=0)               # number predicted as c (column sums)
    actual = matrix.sum(dim=1)                   # number truly c (row sums)
    total = matrix.sum()

    if average == "micro":
        # Pooled counts: micro-P = micro-R = micro-F1 = overall accuracy.
        if total == 0:
            return 0.0, 0.0, 0.0
        acc = (true_positive.sum() / total).item() * 100.0
        return acc, acc, acc

    # Per-class precision/recall with safe division (0 where denominator is 0).
    precision_c = torch.where(predicted > 0, true_positive / predicted, torch.zeros_like(true_positive))
    recall_c = torch.where(actual > 0, true_positive / actual, torch.zeros_like(true_positive))
    denom = precision_c + recall_c
    f1_c = torch.where(denom > 0, 2 * precision_c * recall_c / denom, torch.zeros_like(denom))

    if average == "macro":
        # Unweighted mean over classes.
        precision = precision_c.mean().item()
        recall = recall_c.mean().item()
        f1 = f1_c.mean().item()
    else:  # weighted
        # Mean weighted by class support (true_c / total).
        if total == 0:
            return 0.0, 0.0, 0.0
        weights = actual / total
        precision = (precision_c * weights).sum().item()
        recall = (recall_c * weights).sum().item()
        f1 = (f1_c * weights).sum().item()

    return precision * 100.0, recall * 100.0, f1 * 100.0


class ClassificationMetrics:
    """
    Stateful accumulator for all classification metrics over an epoch.

    Accumulates a confusion matrix and a top-k correct-count across batches,
    plus an optional running mean loss, then derives top-1/top-5 accuracy and
    precision/recall/F1 (for each requested averaging mode) in :meth:`compute`.

    All returned values are percentages in [0, 100].

    Parameters
    ----------
    num_classes:
        Number of classes C.

    topk:
        Tuple of k values for accuracy. Default ``(1, 5)`` -> reports ``top1``
        and ``top5``. The reported keys keep the requested k (so ``top5`` is
        always present when requested); the computation clamps k to
        ``num_classes`` internally, which means ``top5`` is trivially 100% when
        there are fewer than 5 classes.

    averages:
        Averaging modes for precision/recall/F1. Default
        ``("macro", "micro", "weighted")``. The macro values are also exposed
        under the short aliases ``precision``/``recall``/``f1`` for convenient
        use as a monitor metric.
    """

    def __init__(
        self,
        num_classes: int,
        topk: Sequence[int] = (1, 5),
        averages: Sequence[str] = ("macro", "micro", "weighted"),
    ) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be a positive integer.")

        self.num_classes = num_classes
        # Keep the requested k values as the reported labels, but clamp the k
        # actually used in ``topk()`` to the class count (k cannot exceed C).
        self.topk = tuple(topk)
        self._eval_k = {k: min(k, num_classes) for k in self.topk}
        self.averages = tuple(averages)
        for avg in self.averages:
            if avg not in {"macro", "micro", "weighted"}:
                raise ValueError(f"unknown average {avg!r}.")

        self.reset()

    def reset(self) -> None:
        """Zero all accumulators."""
        self._matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.int64)
        self._topk_correct = {k: 0 for k in self.topk}
        self._total = 0
        self._loss = MetricTracker("loss")

    @torch.no_grad()
    def update(
        self,
        logits: Tensor,
        targets: Tensor,
        loss: float | Tensor | None = None,
    ) -> None:
        """
        Accumulate one batch of predictions.

        Parameters
        ----------
        logits:
            Model outputs of shape (B, C).

        targets:
            Ground-truth labels of shape (B,).

        loss:
            Optional scalar loss for this batch, weighted by the batch size in
            the running mean.
        """
        batch_size = targets.shape[0]
        self._total += batch_size

        # Confusion matrix update (for precision/recall/F1).
        self._matrix += confusion_matrix(logits, targets, self.num_classes)

        # Top-k correct counts (so the epoch accuracy is exactly weighted).
        # Use the clamped k for the topk() call (k cannot exceed num_classes).
        max_k = max(self._eval_k.values())
        _, predictions = logits.topk(max_k, dim=1, largest=True, sorted=True)
        predictions = predictions.t()  # (max_k, B)
        correct = predictions.eq(targets.reshape(1, -1).to(predictions.device))
        for k in self.topk:
            self._topk_correct[k] += int(correct[: self._eval_k[k]].reshape(-1).sum().item())

        if loss is not None:
            self._loss.update(loss, n=batch_size)

    def compute(self) -> dict[str, float]:
        """
        Compute all metrics from the accumulated state.

        Returns
        -------
        dict[str, float]
            Keys: ``loss``, ``top1``/``top5`` (per requested k),
            ``precision_<avg>`` / ``recall_<avg>`` / ``f1_<avg>`` for each
            averaging mode, and the macro aliases ``precision``/``recall``/``f1``.
            Accuracy and P/R/F1 are percentages in [0, 100].
        """
        results: dict[str, float] = {"loss": self._loss.avg}

        total = max(self._total, 1)  # guard against division by zero
        for k in self.topk:
            results[f"top{k}"] = 100.0 * self._topk_correct[k] / total

        for avg in self.averages:
            precision, recall, f1 = precision_recall_f1(self._matrix, average=avg)
            results[f"precision_{avg}"] = precision
            results[f"recall_{avg}"] = recall
            results[f"f1_{avg}"] = f1

        # Convenience aliases: bare names map to the macro averages.
        if "macro" in self.averages:
            results["precision"] = results["precision_macro"]
            results["recall"] = results["recall_macro"]
            results["f1"] = results["f1_macro"]

        return results
