"""
Train MobileViT-XS on CIFAR-100.

This script:
1. Downloads CIFAR-100 (if not already present).
2. Build a fixed-scale training DataLoader and a fixed-scale evaluation loader.
3. Builds a MobileViT-XS model with a CIFAR-100-shaped classifier head.
4. Trains for the configured number of epochs with AdamW and cosine LR decay,
   with optional patience-based early stopping on validation top-1.
5. Logs one summary line per epoch (plus a few intra-epoch progress lines)
   to both the console and a log file.
6. Save the trained model weights to disk for later inference.

Example
-------
python scripts/train.py --output mobilevit_xs_cifar100.pt --epochs 100
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Silence torchvision / NumPy 2.4 deprecation noise from the CIFAR-100 reader.
warnings.filterwarnings(
    "ignore",
    message=r".*align should be passed as Python or NumPy boolean.*",
)

import torch
from torch import nn

from data import (
    CIFAR100_NUM_CLASSES,
    build_cifar100_datasets,
    build_eval_transforms,
    build_fixed_scale_loader,
    build_train_transforms,
)
from models.mobilevit import mobilevit_xs
from training import (
    Trainer,
    build_optimizer,
    build_scheduler,
    save_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileViT-XS on CIFAR-100.")
    parser.add_argument("--data-root", default="./cifar100", help="CIFAR-100 download root.")
    parser.add_argument("--output", default="mobilevit_xs_cifar100.pt", help="Output checkpoint path.")
    parser.add_argument("--log-file", default="train.log", help="Path to the training log file.")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of training epochs.")
    parser.add_argument("--image-size", type=int, default=32, help="Spatial size for train and eval. Larger sizes are noticeably slower.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=256, help="Evaluation batch size.")
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
    """
    Configure the root logger to write to both the console and a log file.

    Uses force=True so any previously installed handlers are replaced. The
    FileHandler flushes after every record, so the log file fills in real time.
    """
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


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)
    log = logging.getLogger(__name__)

    log.info(
        f"Run config | epochs {args.epochs} | image_size {args.image_size} | "
        f"batch {args.batch_size} | lr {args.lr} | seed {args.seed}"
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

    log.info("Building MobileViT-XS model...")
    model = mobilevit_xs(num_classes=CIFAR100_NUM_CLASSES)

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
