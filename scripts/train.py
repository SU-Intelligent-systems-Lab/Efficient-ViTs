"""
Universal training script for every model in the Light-weight ViTs library.

One script trains any registered model on ImageNet-100, so all models go through
exactly the same pipeline (same optimizer/scheduler/metric stack) for fair
benchmarking. It:

1. Builds the ImageNet-100 datasets (an ImageFolder tree) and ImageNet
   train/eval transforms.
2. Builds either a fixed-scale loader or a multi-scale loader (multi-scale is
   allowed only for dynamic-input models -- MobileViT and torchvision CNNs).
3. Builds the model via the unified registry.
4. Builds the optimizer, a (warmup +) cosine LR scheduler, and a label-smoothed
   cross-entropy loss.
5. Trains with the unified Trainer (AMP, gradient clipping, all metrics
   including top-1/top-5, throughput logging) and saves only the best weights to
   ``best.pt``.

Examples
--------
python scripts/train.py --model mobilevit_xs --data-root ./imagenet100 \
    --image-size 224 --epochs 100 --batch-size 256 --lr 1e-3 --amp \
    --warmup-epochs 5 --output best_mobilevit_xs.pt

python scripts/train.py --model efficientvit_m0 --data-root ./imagenet100 \
    --image-size 224 --epochs 100 --amp

python scripts/train.py --model mobilevit_xs --data-root ./imagenet100 \
    --image-size 256 --multi-scale --scales 192 224 256
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the package root importable so ``from models import ...`` works no matter
# how this script is invoked (the project root is the parent of scripts/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch import nn

from data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_fixed_scale_loader,
    build_imagenet100k_datasets,
    build_multi_scale_loader,
    make_eval_transform,
    make_train_transform_builder,
)
from models import create_model, get_model_info, list_models, supports_dynamic_input
from training import Trainer, build_optimizer, build_scheduler


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one training run."""
    parser = argparse.ArgumentParser(
        description="Train any registered model on ImageNet-100.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model / dataset
    parser.add_argument("--model", default="mobilevit_xs", help="Registered model name.")
    parser.add_argument("--data-root", default="./imagenet100", help="ImageNet-100 ImageFolder root.")
    parser.add_argument("--num-classes", type=int, default=0, help="0 = infer from dataset.")

    # Schedule / batching
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=224)

    # Optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "adam", "sgd", "rmsprop"])
    parser.add_argument(
        "--scheduler",
        default="cosine",
        choices=["cosine", "step", "multistep", "plateau", "onecycle", "constant"],
    )
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Linear LR warmup (cosine/constant).")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--drop-path", type=float, default=0.0, help="MobileViT only.")
    parser.add_argument("--classifier-dropout", type=float, default=0.0, help="MobileViT only.")

    # Trainer behaviour
    parser.add_argument("--monitor", default="top1", choices=["top1", "top5", "f1", "precision", "recall", "loss"])
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision.")
    parser.add_argument("--grad-clip-norm", type=float, default=0.0, help="0 disables clipping.")
    parser.add_argument("--early-stopping-patience", type=int, default=0, help="0 disables.")

    # Multi-scale sampling
    parser.add_argument("--multi-scale", action="store_true", help="Use the multi-scale sampler.")
    parser.add_argument("--scales", type=int, nargs="+", default=[160, 192, 224, 256])

    # Misc
    parser.add_argument("--output", default="best.pt", help="Best-checkpoint path.")
    parser.add_argument("--log-file", default="train.log")
    parser.add_argument("--device", default=None, help="cuda | cpu (default: auto).")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def setup_logging(log_file: str) -> None:
    """Route logging to both stdout and ``log_file`` (one line per epoch)."""
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
    """End-to-end orchestration: data, model, optimizer, trainer, save best.pt."""
    args = parse_args()
    setup_logging(args.log_file)
    log = logging.getLogger(__name__)

    if args.model not in list_models():
        raise SystemExit(f"Unknown model {args.model!r}. Available:\n{list_models()}")

    torch.manual_seed(args.seed)

    log.info("Loading ImageNet-100 from %s ...", args.data_root)
    train_ds, val_ds = build_imagenet100k_datasets(args.data_root)
    dataset_classes = len(train_ds.classes)
    num_classes = args.num_classes if args.num_classes > 0 else dataset_classes
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    log.info("Dataset ready | train %d | val %d | classes %d", len(train_ds), len(val_ds), num_classes)

    # ---- Train loader: fixed-scale or multi-scale ----
    use_multi_scale = args.multi_scale
    if use_multi_scale and not supports_dynamic_input(args.model):
        # Fixed-resolution models (EfficientViT / EfficientFormer) cannot take
        # variable input sizes; fall back to fixed-scale with a clear warning.
        log.warning(
            "Model %s is fixed-resolution and does not support multi-scale "
            "sampling; using fixed-scale training at %d.",
            args.model,
            args.image_size,
        )
        use_multi_scale = False

    if use_multi_scale:
        log.info("Building multi-scale train loader | scales %s | base %d", args.scales, args.image_size)
        train_loader = build_multi_scale_loader(
            train_ds,
            transform_builder=make_train_transform_builder(mean=mean, std=std),
            scales=tuple(args.scales),
            base_size=args.image_size,
            base_batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )
    else:
        train_loader = build_fixed_scale_loader(
            train_ds,
            transform=make_train_transform_builder(mean=mean, std=std)(args.image_size),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
        )

    eval_loader = build_fixed_scale_loader(
        val_ds,
        transform=make_eval_transform(args.image_size, mean=mean, std=std),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # ---- Model ----
    # Only MobileViT accepts drop_path / classifier_dropout; pass them only there.
    model_kwargs: dict[str, float] = {}
    if get_model_info(args.model).family == "mobilevit":
        model_kwargs["drop_path"] = args.drop_path
        model_kwargs["classifier_dropout"] = args.classifier_dropout

    log.info("Building model %s | classes %d | image_size %d", args.model, num_classes, args.image_size)
    model = create_model(
        args.model,
        num_classes=num_classes,
        in_channels=3,
        image_size=args.image_size,
        **model_kwargs,
    )

    # ---- Optimizer / scheduler / loss ----
    optimizer = build_optimizer(
        model, name=args.optimizer, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = build_scheduler(
        optimizer,
        args.scheduler,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        # ReduceLROnPlateau direction follows the monitor: loss -> min else max.
        plateau_mode="min" if args.monitor == "loss" else "max",
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=eval_loader,
        optimizer=optimizer,
        num_classes=num_classes,
        criterion=criterion,
        scheduler=scheduler,
        device=args.device,
        monitor=args.monitor,
        amp=args.amp,
        grad_clip_norm=args.grad_clip_norm if args.grad_clip_norm > 0 else None,
        checkpoint_path=args.output,
        early_stopping_patience=args.early_stopping_patience or None,
    )

    trainer.train(epochs=args.epochs)
    log.info("Training complete. Best %s = %.4f. Best weights at %s.",
             args.monitor, trainer.best_metric, args.output)


if __name__ == "__main__":
    main()
