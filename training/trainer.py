"""
Unified trainer for every model in the Light-weight ViTs library.

This ``Trainer`` drives a standard image-classification training/validation loop
for any model that satisfies the library's contract (MobileViT, EfficientViT,
EfficientFormer, torchvision CNNs). Using one trainer with one optimizer/
scheduler/metric stack is what makes cross-model benchmarking fair: every model
is trained the same way.

What it does each epoch
-----------------------
1. Train: forward, loss, backward, (optional) gradient clipping, optimizer step,
   per-iteration scheduler step. Metrics accumulate in a
   :class:`~training.metrics.ClassificationMetrics`.
2. Validate (if a ``val_loader`` is given): the same metrics on the held-out set.
3. Step any per-epoch scheduler (passing the monitored metric to
   ReduceLROnPlateau).
4. Update the best checkpoint: if the monitored validation metric improves, save
   the model ``state_dict`` to ``best.pt`` (and only ``best.pt`` -- see
   :mod:`training.checkpoint`).
5. Log one summary line: epoch, time, throughput (images/s), learning rate,
   train and validation metrics, and the running best.

Mixed precision and gradient clipping
--------------------------------------
``amp=True`` runs the forward/backward under ``torch.autocast`` (float16 on CUDA,
bfloat16 on CPU) with a ``GradScaler`` on CUDA. ``grad_clip_norm`` clips the
global gradient norm after unscaling.

Throughput
----------
Throughput is reported as images per second over the training phase of the
epoch: ``images_processed / train_seconds``.

Best-model monitoring
---------------------
``monitor`` names the validation metric used to define "best" (e.g. ``"top1"``
or ``"f1"``); ``monitor_mode`` is ``"max"`` for accuracy/F1 and ``"min"`` for
loss (auto-derived from the name when not given). Only the best weights are
written to ``checkpoint_path``.

Logging is emitted on the module logger; the calling script configures handlers.

Example
-------
>>> import logging
>>> from training import Trainer, build_optimizer, build_scheduler
>>>
>>> logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
>>> optimizer = build_optimizer(model, name="adamw", lr=1e-3)
>>> spec = build_scheduler(optimizer, "cosine", epochs=50, steps_per_epoch=len(train_loader))
>>> trainer = Trainer(
...     model=model, train_loader=train_loader, val_loader=val_loader,
...     optimizer=optimizer, scheduler=spec, num_classes=100,
...     monitor="top1", amp=True, checkpoint_path="best.pt",
... )
>>> history = trainer.train(epochs=50)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .checkpoint import save_model
from .metrics import ClassificationMetrics
from .optimizer import SchedulerSpec


__all__ = ["Trainer"]


logger = logging.getLogger(__name__)

# Metrics for which a smaller value is better (everything else is "higher is
# better"). Used to auto-pick the monitor direction.
_LOWER_IS_BETTER = {"loss"}


class Trainer:
    """
    Compact, model-agnostic trainer for image classification.

    Parameters
    ----------
    model:
        Model to train (any model satisfying the unified contract).

    train_loader:
        Training dataloader yielding ``(images, targets)``.

    optimizer:
        Optimizer used for training.

    num_classes:
        Number of classes; required to configure the metric accumulator.

    criterion:
        Loss function. Defaults to ``nn.CrossEntropyLoss()``.

    val_loader:
        Optional validation dataloader.

    scheduler:
        Optional LR scheduler. Either a :class:`~training.optimizer.SchedulerSpec`
        (preferred -- carries stepping metadata) or a bare ``LRScheduler`` (then
        assumed to be stepped once per optimizer update).

    device:
        Training device. If None, CUDA when available else CPU.

    early_stopping_patience:
        Stop after this many epochs without monitor improvement. Requires a
        ``val_loader``. None or 0 disables early stopping.

    monitor:
        Validation metric key used for best-model selection and early stopping
        (e.g. ``"top1"``, ``"top5"``, ``"f1"``, ``"loss"``). Default ``"top1"``.

    monitor_mode:
        ``"max"`` or ``"min"``. When None, derived from ``monitor``
        (``"loss"`` -> "min", otherwise "max").

    amp:
        Enable automatic mixed precision.

    grad_clip_norm:
        Max global gradient norm for clipping (None / 0 disables).

    checkpoint_path:
        Where to write ``best.pt``. If None, no checkpoint is written.

    topk:
        Top-k values reported (default ``(1, 5)``).

    metric_averages:
        Averaging modes for precision/recall/F1 (default macro/micro/weighted).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        optimizer: Optimizer,
        num_classes: int,
        criterion: nn.Module | None = None,
        val_loader=None,
        scheduler: SchedulerSpec | LRScheduler | None = None,
        device: str | torch.device | None = None,
        early_stopping_patience: int | None = None,
        monitor: str = "top1",
        monitor_mode: str | None = None,
        amp: bool = False,
        grad_clip_norm: float | None = None,
        checkpoint_path: str | Path | None = None,
        topk: Sequence[int] = (1, 5),
        metric_averages: Sequence[str] = ("macro", "micro", "weighted"),
    ) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be a positive integer.")
        if early_stopping_patience is not None and early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative or None.")

        self.device = torch.device(device) if device is not None else self._default_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.topk = tuple(topk)
        self.metric_averages = tuple(metric_averages)

        # Normalize the scheduler into a SchedulerSpec (bare schedulers are
        # assumed to step every optimizer update).
        if scheduler is None:
            self.scheduler_spec: SchedulerSpec | None = None
        elif isinstance(scheduler, SchedulerSpec):
            self.scheduler_spec = scheduler
        else:
            self.scheduler_spec = SchedulerSpec(scheduler, interval="step")

        # 0 means "disabled" for both knobs.
        self.early_stopping_patience = (
            early_stopping_patience if early_stopping_patience else None
        )

        self.monitor = monitor
        self.monitor_mode = monitor_mode or ("min" if monitor in _LOWER_IS_BETTER else "max")
        if self.monitor_mode not in {"max", "min"}:
            raise ValueError("monitor_mode must be 'max' or 'min'.")

        self.amp = amp
        # GradScaler only does work on CUDA float16; on CPU autocast uses
        # bfloat16 and needs no scaling.
        self.scaler = torch.amp.GradScaler(enabled=amp and self.device.type == "cuda")
        self.grad_clip_norm = grad_clip_norm if grad_clip_norm else None
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None

        # Best monitored value seen so far; start at the worst possible value.
        self.best_metric = float("-inf") if self.monitor_mode == "max" else float("inf")

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _default_device() -> torch.device:
        """Return CUDA when available, otherwise CPU."""
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    @staticmethod
    def _resolve_logits(output) -> Tensor:
        """
        Reduce a model output to a single logits tensor.

        Most models return a tensor. EfficientViT with ``distillation=True``
        returns a ``(logits, dist_logits)`` tuple during training; we average
        the two heads (an ensemble) so the loss and metrics see one tensor.
        """
        if torch.is_tensor(output):
            return output
        # Tuple/list of head logits -> average.
        return sum(output) / len(output)

    def _advance_sampler_epoch(self, epoch: int) -> None:
        """
        Notify the train loader's sampler of the new epoch.

        Samplers that implement ``set_epoch`` (e.g. ``MultiScaleSampler``) use it
        to re-derive their per-epoch shuffling. Plain loaders are unaffected.
        """
        sampler = getattr(self.train_loader, "batch_sampler", None)
        if sampler is None:
            sampler = getattr(self.train_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    def _is_improvement(self, value: float) -> bool:
        """Return True if ``value`` beats the current best under monitor_mode."""
        if self.monitor_mode == "max":
            return value > self.best_metric
        return value < self.best_metric

    def _current_lr(self) -> float:
        """Return the learning rate of the optimizer's first parameter group."""
        return self.optimizer.param_groups[0]["lr"]

    # --------------------------------------------------------------- training

    def train(self, epochs: int) -> list[dict[str, float]]:
        """
        Train for at most ``epochs`` epochs.

        Returns
        -------
        list[dict[str, float]]
            One record per epoch. Each record contains every train/validation
            metric (prefixed ``train_`` / ``val_``) plus ``lr``, ``epoch_time``,
            and ``images_per_sec``.
        """
        if epochs <= 0:
            raise ValueError("epochs must be a positive integer.")

        logger.info(
            "Training on %s | monitor %s (%s) | amp %s | grad_clip %s | checkpoint %s",
            self.device,
            self.monitor,
            self.monitor_mode,
            self.amp,
            self.grad_clip_norm,
            self.checkpoint_path if self.checkpoint_path is not None else "(none)",
        )

        history: list[dict[str, float]] = []
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()

            train_metrics, images_per_sec = self.train_one_epoch(epoch)

            val_metrics: dict[str, float] | None = None
            if self.val_loader is not None:
                val_metrics = self.evaluate()

            # Source of the monitored value: validation if available, else train.
            monitor_source = val_metrics if val_metrics is not None else train_metrics
            monitor_value = monitor_source.get(self.monitor)

            # Per-epoch scheduler stepping (after metrics are known).
            self._step_epoch_scheduler(monitor_value)

            # Best-model checkpointing + early-stopping bookkeeping.
            improved = False
            if monitor_value is not None and self._is_improvement(monitor_value):
                self.best_metric = monitor_value
                improved = True
                epochs_without_improvement = 0
                self._save_best(epoch, monitor_value)
            else:
                epochs_without_improvement += 1

            self._log_epoch(
                epoch=epoch,
                total_epochs=epochs,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                epoch_start=epoch_start,
                images_per_sec=images_per_sec,
                improved=improved,
            )

            # Assemble the history record.
            record: dict[str, float] = {f"train_{k}": v for k, v in train_metrics.items()}
            if val_metrics is not None:
                record.update({f"val_{k}": v for k, v in val_metrics.items()})
            record["lr"] = self._current_lr()
            record["epoch_time"] = time.perf_counter() - epoch_start
            record["images_per_sec"] = images_per_sec
            history.append(record)

            if (
                self.early_stopping_patience is not None
                and self.val_loader is not None
                and epochs_without_improvement >= self.early_stopping_patience
            ):
                logger.info(
                    "Early stopping after epoch %d: no %s improvement for %d epochs "
                    "(best %s %.4f).",
                    epoch,
                    self.monitor,
                    self.early_stopping_patience,
                    self.monitor,
                    self.best_metric,
                )
                break

        return history

    def train_one_epoch(self, epoch: int) -> tuple[dict[str, float], float]:
        """
        Train for one epoch.

        Returns
        -------
        metrics:
            The full metric dict for the epoch (loss, top-k, P/R/F1).

        images_per_sec:
            Training throughput over the epoch.
        """
        self._advance_sampler_epoch(epoch)
        self.model.train()

        metrics = ClassificationMetrics(self.num_classes, self.topk, self.metric_averages)
        total_images = 0
        start = time.perf_counter()

        for images, targets in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            batch_size = images.shape[0]
            total_images += batch_size

            self.optimizer.zero_grad(set_to_none=True)

            # Forward + loss under autocast when AMP is enabled.
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                output = self.model(images)
                logits = self._resolve_logits(output)
                loss = self.criterion(logits, targets)

            # Backward with optional gradient scaling (CUDA float16 AMP).
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                if self.grad_clip_norm is not None:
                    # Unscale before clipping so the norm is in real units.
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            # Per-iteration scheduler step (cosine / onecycle).
            if (
                self.scheduler_spec is not None
                and self.scheduler_spec.scheduler is not None
                and self.scheduler_spec.interval == "step"
            ):
                self.scheduler_spec.scheduler.step()

            # Metrics use the (detached) float32 logits.
            metrics.update(logits.detach().float(), targets, loss=loss)

        elapsed = max(time.perf_counter() - start, 1e-9)
        images_per_sec = total_images / elapsed
        return metrics.compute(), images_per_sec

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """
        Evaluate the model on the validation dataloader.

        Returns
        -------
        dict[str, float]
            The full metric dict (loss, top-k, precision/recall/F1).
        """
        if self.val_loader is None:
            raise ValueError("val_loader is not set.")

        self.model.eval()
        metrics = ClassificationMetrics(self.num_classes, self.topk, self.metric_averages)

        for images, targets in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                output = self.model(images)
                logits = self._resolve_logits(output)
                loss = self.criterion(logits, targets)

            metrics.update(logits.float(), targets, loss=loss)

        return metrics.compute()

    # ------------------------------------------------------------- scheduler

    def _step_epoch_scheduler(self, monitor_value: float | None) -> None:
        """Step a per-epoch scheduler, passing the metric to ReduceLROnPlateau."""
        spec = self.scheduler_spec
        if spec is None or spec.scheduler is None or spec.interval != "epoch":
            return
        if spec.needs_metric:
            # ReduceLROnPlateau needs the monitored metric; skip if unavailable.
            if monitor_value is not None:
                spec.scheduler.step(monitor_value)
        else:
            spec.scheduler.step()

    # --------------------------------------------------------- checkpointing

    def _save_best(self, epoch: int, monitor_value: float) -> None:
        """Write the current weights to ``best.pt`` and log it."""
        if self.checkpoint_path is None:
            return
        save_model(self.checkpoint_path, self.model)
        logger.info(
            "  -> saved best.pt at epoch %d (%s %.4f) to %s",
            epoch,
            self.monitor,
            monitor_value,
            self.checkpoint_path,
        )

    # -------------------------------------------------------------- logging

    def _log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
        epoch_start: float,
        images_per_sec: float,
        improved: bool,
    ) -> None:
        """Emit one readable summary line for the epoch."""
        elapsed = time.perf_counter() - epoch_start

        def fmt(metrics: dict[str, float]) -> str:
            # Headline of each metric family: loss, top-1/5, macro P/R/F1.
            return (
                f"loss {metrics['loss']:.4f} "
                f"top1 {metrics.get('top1', 0.0):6.2f} "
                f"top5 {metrics.get('top5', 0.0):6.2f} "
                f"P {metrics.get('precision', 0.0):6.2f} "
                f"R {metrics.get('recall', 0.0):6.2f} "
                f"F1 {metrics.get('f1', 0.0):6.2f}"
            )

        parts = [
            f"Epoch {epoch:3d}/{total_epochs}",
            f"time {elapsed:6.1f}s",
            f"{images_per_sec:7.1f} img/s",
            f"lr {self._current_lr():.2e}",
            f"train [{fmt(train_metrics)}]",
        ]
        if val_metrics is not None:
            parts.append(f"val [{fmt(val_metrics)}]")
        best_marker = " *" if improved else ""
        parts.append(f"best {self.monitor} {self.best_metric:6.2f}{best_marker}")

        logger.info(" | ".join(parts))
