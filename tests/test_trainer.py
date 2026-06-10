"""One-epoch smoke tests for the unified Trainer."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from models import create_model
from training import Trainer, build_optimizer, build_scheduler, load_model


def _random_loader(num_samples=24, num_classes=5, size=32, batch_size=8):
    images = torch.randn(num_samples, 3, size, size)
    targets = torch.randint(0, num_classes, (num_samples,))
    return DataLoader(TensorDataset(images, targets), batch_size=batch_size)


def test_trainer_one_epoch_saves_best(tmp_path):
    num_classes = 5
    train_loader = _random_loader(num_classes=num_classes)
    val_loader = _random_loader(num_classes=num_classes)

    model = create_model("mobilevit_xxs", num_classes=num_classes)
    optimizer = build_optimizer(model, "adamw", lr=1e-3)
    scheduler = build_scheduler(optimizer, "cosine", epochs=2, steps_per_epoch=len(train_loader))

    ckpt = tmp_path / "best.pt"
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_classes=num_classes,
        scheduler=scheduler,
        device="cpu",
        monitor="top1",
        checkpoint_path=ckpt,
    )

    history = trainer.train(epochs=2)

    # History has one record per epoch with prefixed metrics + extras.
    assert len(history) == 2
    record = history[0]
    for key in ("train_loss", "train_top1", "val_top1", "val_f1", "lr", "images_per_sec"):
        assert key in record

    # best.pt was written and reloads into a fresh model.
    assert ckpt.exists()
    fresh = create_model("mobilevit_xxs", num_classes=num_classes)
    load_model(ckpt, fresh, device="cpu")


def test_trainer_monitor_loss_is_min_mode():
    train_loader = _random_loader()
    model = create_model("mobilevit_xxs", num_classes=5)
    optimizer = build_optimizer(model, "sgd", lr=1e-2)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        num_classes=5,
        device="cpu",
        monitor="loss",
    )
    # "loss" auto-selects min mode; best starts at +inf.
    assert trainer.monitor_mode == "min"
    assert trainer.best_metric == float("inf")

    trainer.train(epochs=1)
    # After training, a finite best loss has been recorded.
    assert trainer.best_metric < float("inf")


def test_trainer_amp_cpu_runs():
    # AMP path should at least run on CPU (bfloat16 autocast, no GradScaler).
    train_loader = _random_loader()
    model = create_model("mobilevit_xxs", num_classes=5)
    optimizer = build_optimizer(model, "adamw", lr=1e-3)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        num_classes=5,
        device="cpu",
        amp=True,
        grad_clip_norm=1.0,
    )
    history = trainer.train(epochs=1)
    assert len(history) == 1
