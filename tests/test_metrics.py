"""Tests for accuracy, precision/recall/F1, and the metric accumulator."""

import torch

from training.metrics import (
    ClassificationMetrics,
    accuracy,
    confusion_matrix,
    precision_recall_f1,
)


def test_accuracy_perfect_and_zero():
    # Logits that put all mass on the true class -> top1 = 100%.
    targets = torch.tensor([0, 1, 2, 3])
    logits = torch.eye(4)[targets] * 10.0
    top1 = accuracy(logits, targets, topk=(1,))[0]
    assert top1.item() == 100.0

    # Always predict class 0 -> only the sample whose target is 0 is correct.
    logits_zero = torch.zeros(4, 4)
    logits_zero[:, 0] = 10.0
    top1_zero = accuracy(logits_zero, targets, topk=(1,))[0]
    assert top1_zero.item() == 25.0


def test_accuracy_topk_monotonic():
    torch.manual_seed(0)
    logits = torch.randn(32, 10)
    targets = torch.randint(0, 10, (32,))
    top1, top5 = accuracy(logits, targets, topk=(1, 5))
    # top-5 accuracy is never lower than top-1.
    assert top5.item() >= top1.item()


def test_confusion_matrix_counts():
    targets = torch.tensor([0, 0, 1, 1])
    preds = torch.tensor([0, 1, 1, 1])
    cm = confusion_matrix(preds, targets, num_classes=2)
    # Row = true class, column = predicted class.
    assert cm.tolist() == [[1, 1], [0, 2]]


def test_precision_recall_f1_known_matrix():
    # Confusion matrix [[5, 1], [2, 2]] (rows = true, cols = pred).
    matrix = torch.tensor([[5, 1], [2, 2]])

    # micro precision = recall = f1 = accuracy = (5 + 2) / 10 = 70%.
    p_mi, r_mi, f_mi = precision_recall_f1(matrix, average="micro")
    assert abs(p_mi - 70.0) < 1e-6
    assert p_mi == r_mi == f_mi

    # macro F1 = mean(0.76923, 0.57143) * 100 ~= 67.03.
    _, _, f_ma = precision_recall_f1(matrix, average="macro")
    assert abs(f_ma - 67.03) < 0.1

    # weighted recall = 0.6*0.833 + 0.4*0.5 = 0.70 -> 70%.
    _, r_w, _ = precision_recall_f1(matrix, average="weighted")
    assert abs(r_w - 70.0) < 0.1


def test_classification_metrics_keys_and_perfect():
    targets = torch.arange(10)
    logits = torch.eye(10)[targets] * 10.0
    metrics = ClassificationMetrics(num_classes=10)
    metrics.update(logits, targets, loss=0.25)
    out = metrics.compute()

    # All metric families are present, including aliases.
    for key in ("loss", "top1", "top5", "precision_macro", "recall_micro", "f1_weighted", "f1"):
        assert key in out

    assert out["loss"] == 0.25
    assert out["top1"] == 100.0
    # Every class present exactly once and predicted correctly -> macro F1 = 100.
    assert abs(out["f1_macro"] - 100.0) < 1e-6
    assert abs(out["precision_micro"] - 100.0) < 1e-6


def test_classification_metrics_reports_requested_topk_labels():
    # The requested labels are kept (top5 present), but with only 3 classes the
    # top-5 computation is clamped to top-3, which is trivially 100% correct.
    metrics = ClassificationMetrics(num_classes=3, topk=(1, 5))
    assert metrics.topk == (1, 5)

    targets = torch.tensor([0, 1, 2, 0])
    logits = torch.randn(4, 3)
    metrics.update(logits, targets)
    out = metrics.compute()
    assert "top5" in out
    assert out["top5"] == 100.0  # k >= num_classes -> always correct
