"""
Train torchvision's MobileNetV2 on CIFAR-100.

Mirrors scripts/train.py (same batch size, image size, epochs, early stopping,
optimizer, scheduler, etc.) but swaps the model for torchvision's stock
MobileNetV2, with its classifier head resized to CIFAR-100's class count.
Writes to a separate log file by default.

Example
-------
python scripts/train_mobilenetv2.py --output mobilenetv2_cifar100.pt --epochs 100
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings(
    "ignore",
    message=r".*align should be passed as Python or NumPy boolean.*",
)

import torch
from torch import nn
from torchvision.models import mobilenet_v2

from data import (
    CIFAR100_NUM_CLASSES,
    build_cifar100_datasets,
    build_eval_transforms,
    build_fixed_scale_loader,
    build_train_transforms,
)
from training import (
    Trainer,
    build_optimizer,
    build_scheduler,
    save_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train torchvision MobileNetV2 on CIFAR-100.")
    parser.add_argument("--data-root", default="./cifar100", help="CIFAR-100 download root.")
    parser.add_argument("--output", default="mobilenetv2_cifar100.pt", help="Output checkpoint path.")
    parser.add_argument("--log-file", default="train_mobilenetv2.log", help="Path to the training log file.")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum number of training epochs.")
    parser.add_argument("--image-size", type=int, default=128, help="Spatial size for train and eval.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay.")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Stop after this many epochs without val top-1 improvement. 0 disables.",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--device", default=None, help="Device, e.g. 'cuda' or 'cpu'. Default: auto.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def setup_logging(log_file: str) -> None:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def build_mobilenetv2(num_classes: int) -> nn.Module:
    model = mobilenet_v2(weights=None, num_classes=num_classes)
    return model


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)
    log = logging.getLogger(__name__)

    log.info(
        f"Run config | model MobileNetV2 (torchvision) | epochs {args.epochs} | "
        f"image_size {args.image_size} | batch {args.batch_size} | lr {args.lr} | seed {args.seed}"
    )

    torch.manual_seed(args.seed)

    log.info(f"Loading CIFAR-100 from {args.data_root} (download if missing)...")
    train_dataset, test_dataset = build_cifar100_datasets(
        root=args.data_root,
        download=True,
    )
    log.info(f"CIFAR-100 ready | train {len(train_dataset)} | test {len(test_dataset)}")

    log.info("Building dataloaders...")
    train_loader = build_fixed_scale_loader(
        train_dataset,
        transform=build_train_transforms(size=args.image_size),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    eval_loader = build_fixed_scale_loader(
        test_dataset,
        transform=build_eval_transforms(size=args.image_size),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    log.info("Building torchvision MobileNetV2 model...")
    model = build_mobilenetv2(num_classes=CIFAR100_NUM_CLASSES)

    optimizer = build_optimizer(
        model,
        name="adamw",
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
    )
    criterion = nn.CrossEntropyLoss()

    patience = args.early_stopping_patience if args.early_stopping_patience > 0 else None

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=eval_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=args.device,
        early_stopping_patience=patience,
    )

    log.info(
        f"Starting training | device {trainer.device} | "
        f"early-stopping patience {patience}"
    )

    trainer.train(epochs=args.epochs)

    save_model(args.output, model)
    log.info(f"Saved trained model weights to {args.output}.")


if __name__ == "__main__":
    main()
