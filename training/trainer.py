"""
Simple trainer for MobileViT models.

This module provides Trainer: a compact training and validation loop that
emits records via the standard logging module:
1. One line when each epoch starts.
2. A handful of intra-epoch progress lines (roughly every 10% of the batches).
3. One summary line at the end of each epoch.

Each epoch summary reports the wall time, training loss / top-1 / top-5,
validation loss / top-1 / top-5 (when a val_loader is provided), and the
best training and validation top-1 accuracy seen so far.

If early_stopping_patience is set, training stops after that many consecutive
epochs without an improvement in validation top-1.

Logging is emitted on the logger named after this module. The trainer does
not configure handlers; the calling application is responsible for routing
records to the console, a file, or anywhere else.

Example
-------
>>> import logging
>>> from mobilevit.training.trainer import Trainer
>>>
>>> logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
>>>
>>> trainer = Trainer(
...     model=model,
...     train_loader=train_loader,
...     val_loader=val_loader,
...     optimizer=optimizer,
...     scheduler=scheduler,
...     early_stopping_patience=10,
... )
>>>
>>> history = trainer.train(epochs=100)
"""

from __future__ import annotations

import logging
import time

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from .metrics import MetricTracker, accuracy


__all__ = ["Trainer"]


logger = logging.getLogger(__name__)


class Trainer:
    """
    Compact trainer for image classification.

    Parameters
    ----------
    model:
        Model to train.

    train_loader:
        Training dataloader.

    optimizer:
        Optimizer used for training.

    criterion:
        Loss function. Defaults to nn.CrossEntropyLoss().

    val_loader:
        Optional validation dataloader.

    scheduler:
        Optional learning-rate scheduler. Stepped once per optimizer update.

    device:
        Training device. If None, CUDA is used when available, otherwise CPU.

    early_stopping_patience:
        If set to a positive integer, training stops after that many
        consecutive epochs without an improvement in validation top-1.
        Requires a val_loader. If None, early stopping is disabled.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module | None = None,
        val_loader: DataLoader | None = None,
        scheduler: LRScheduler | None = None,
        device: str | torch.device | None = None,
        early_stopping_patience: int | None = None,
    ) -> None:
        if early_stopping_patience is not None and early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be a positive integer or None.")

        self.device = (
            torch.device(device) if device is not None else self._default_device()
        )

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = (
            criterion if criterion is not None else nn.CrossEntropyLoss()
        )
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience

        self.best_train_top1 = 0.0
        self.best_val_top1 = 0.0

    @staticmethod
    def _default_device() -> torch.device:
        """Return CUDA when available, otherwise CPU."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _advance_sampler_epoch(self, epoch: int) -> None:
        """
        Notify the train loader's sampler of the new epoch.

        Samplers that implement set_epoch (e.g. MultiScaleSampler) use it to
        re-derive their per-epoch shuffling. Plain DataLoaders without such a
        sampler are unaffected.
        """
        sampler = getattr(self.train_loader, "batch_sampler", None)
        if sampler is None:
            sampler = getattr(self.train_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    def train(self, epochs: int) -> list[dict[str, float]]:
        """
        Train the model for at most ``epochs`` epochs.

        Stops early if early_stopping_patience is set and validation top-1
        has not improved for that many epochs in a row.

        Returns
        -------
        list[dict[str, float]]
            One dictionary per completed epoch with train and validation metrics.
        """
        if epochs <= 0:
            raise ValueError("epochs must be a positive integer.")

        history: list[dict[str, float]] = []
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()

            train_metrics = self.train_one_epoch(epoch, total_epochs=epochs)
            self.best_train_top1 = max(self.best_train_top1, train_metrics["top1"])

            val_metrics: dict[str, float] | None = None
            if self.val_loader is not None:
                val_metrics = self.evaluate()
                if val_metrics["top1"] > self.best_val_top1:
                    self.best_val_top1 = val_metrics["top1"]
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            self._log_epoch(
                epoch=epoch,
                total_epochs=epochs,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                epoch_start=epoch_start,
            )

            record = {f"train_{key}": value for key, value in train_metrics.items()}
            if val_metrics is not None:
                record.update({f"val_{key}": value for key, value in val_metrics.items()})
            history.append(record)

            if (
                self.early_stopping_patience is not None
                and self.val_loader is not None
                and epochs_without_improvement >= self.early_stopping_patience
            ):
                logger.info(
                    f"Early stopping after epoch {epoch}: "
                    f"no validation top-1 improvement for "
                    f"{self.early_stopping_patience} epochs "
                    f"(best val top1 {self.best_val_top1:.2f})."
                )
                break

        return history

    def train_one_epoch(
        self,
        epoch: int,
        total_epochs: int | None = None,
    ) -> dict[str, float]:
        """
        Train for one epoch.

        Silent loop: progress is reported only by the epoch summary emitted
        from train().
        """
        del total_epochs  # only used by the caller for logging.

        self._advance_sampler_epoch(epoch)
        self.model.train()

        loss_tracker = MetricTracker("loss")
        top1_tracker = MetricTracker("top1")
        top5_tracker = MetricTracker("top5")

        for batch in self.train_loader:
            images, targets = batch
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            batch_size = images.shape[0]

            logits = self.model(images)
            loss = self.criterion(logits, targets)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            top5_k = min(5, logits.shape[1])
            top1, top5 = accuracy(logits, targets, topk=(1, top5_k))

            loss_tracker.update(loss, n=batch_size)
            top1_tracker.update(top1, n=batch_size)
            top5_tracker.update(top5, n=batch_size)

        return {
            "loss": loss_tracker.avg,
            "top1": top1_tracker.avg,
            "top5": top5_tracker.avg,
        }

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """
        Evaluate the model on the validation dataloader.
        """
        if self.val_loader is None:
            raise ValueError("val_loader is not set.")

        self.model.eval()

        loss_tracker = MetricTracker("loss")
        top1_tracker = MetricTracker("top1")
        top5_tracker = MetricTracker("top5")

        for batch in self.val_loader:
            images, targets = batch
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            batch_size = images.shape[0]

            logits = self.model(images)
            loss = self.criterion(logits, targets)
            top5_k = min(5, logits.shape[1])
            top1, top5 = accuracy(logits, targets, topk=(1, top5_k))

            loss_tracker.update(loss, n=batch_size)
            top1_tracker.update(top1, n=batch_size)
            top5_tracker.update(top5, n=batch_size)

        return {
            "loss": loss_tracker.avg,
            "top1": top1_tracker.avg,
            "top5": top5_tracker.avg,
        }

    def _log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
        epoch_start: float,
    ) -> None:
        """
        Emit one log line summarizing the epoch.
        """
        elapsed = time.perf_counter() - epoch_start

        parts = [
            f"Epoch {epoch:3d}/{total_epochs}",
            f"time {elapsed:6.1f}s",
            f"train loss {train_metrics['loss']:.4f}",
            f"top1 {train_metrics['top1']:6.2f}",
            f"top5 {train_metrics['top5']:6.2f}",
        ]
        if val_metrics is not None:
            parts.extend(
                [
                    f"val loss {val_metrics['loss']:.4f}",
                    f"top1 {val_metrics['top1']:6.2f}",
                    f"top5 {val_metrics['top5']:6.2f}",
                ]
            )

        parts.append(f"best train {self.best_train_top1:6.2f}")
        if self.val_loader is not None:
            parts.append(f"best val {self.best_val_top1:6.2f}")

        logger.info(" | ".join(parts))
