"""
Run inference with a trained MobileViT-XS model on a single image.

This script loads the model weights saved by scripts/train.py, runs the image
through the fixed-scale evaluation pipeline, and prints the top-k predictions.

Example
-------
python scripts/infer.py --checkpoint mobilevit_xs_cifar100.pt --image cat.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from data import CIFAR100_NUM_CLASSES
from inference import Predictor
from models.mobilevit import mobilevit_xs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with MobileViT-XS.")
    parser.add_argument("--checkpoint", required=True, help="Path to saved model weights.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--size", type=int, default=256, help="Spatial size used for inference.")
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions to print.")
    parser.add_argument("--device", default=None, help="Device, e.g. 'cuda' or 'cpu'. Default: auto.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = mobilevit_xs(num_classes=CIFAR100_NUM_CLASSES)

    predictor = Predictor.from_checkpoint(
        path=args.checkpoint,
        model=model,
        size=args.size,
        device=args.device,
    )

    image = Image.open(args.image).convert("RGB")
    probs, indices = predictor.predict(image, topk=args.topk)

    print(f"Top-{args.topk} predictions for {args.image}:")
    for rank, (prob, idx) in enumerate(
        zip(probs.tolist(), indices.tolist()),
        start=1,
    ):
        print(f"  {rank}. class {idx:3d}  prob {prob:.4f}")


if __name__ == "__main__":
    main()
