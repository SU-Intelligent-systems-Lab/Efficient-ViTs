"""
Run inference with a trained model on a single image.

This script is a thin CLI wrapper around :class:`inference.predictor.Predictor`:
it only parses arguments, delegates all model building / weight loading /
pre-processing to the Predictor, and prints the top-k predictions. All reusable
logic lives in the Predictor so it can be used as a library API too.

Because every model is built through the unified registry, the same script
serves all model families -- just change ``--model``.

Two weight sources:
- ``--pretrained``: load the bundled local weights (``weights/<model>.pt``, the
  ImageNet-100 checkpoints trained with this library) -- no ``--checkpoint`` or
  ``--num-classes`` needed.
- ``--checkpoint PATH``: load an explicit ``state_dict`` file (then
  ``--num-classes`` is required).

Examples
--------
python scripts/infer.py --model mobilevit_xs --pretrained --image cat.jpg --topk 5

python scripts/infer.py --model mobilevit_xs --checkpoint best.pt \
    --num-classes 100 --image cat.jpg --image-size 224 --topk 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package root importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from inference import Predictor


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one inference run."""
    parser = argparse.ArgumentParser(
        description="Run inference with a trained model on a single image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Registered model name (must match weights).")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Load the bundled local weights (weights/<model>.pt). No checkpoint needed.",
    )
    parser.add_argument("--checkpoint", default=None, help="Path to a state_dict file (if not --pretrained).")
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of classes (required with --checkpoint; inferred with --pretrained).",
    )
    parser.add_argument("--image-size", type=int, default=None, help="Inference size (default: model's training size).")
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions to print.")
    parser.add_argument("--device", default=None, help="cuda | cpu (default: auto).")
    return parser.parse_args()


def main() -> None:
    """Build the predictor (pretrained or checkpoint), classify, print top-k."""
    args = parse_args()

    if args.pretrained:
        # Bundled local weights; num_classes/image_size default to the trained values.
        predictor = Predictor.from_pretrained(
            model_name=args.model,
            num_classes=args.num_classes,
            image_size=args.image_size,
            device=args.device,
        )
    else:
        if args.checkpoint is None or args.num_classes is None:
            raise SystemExit(
                "Without --pretrained you must pass both --checkpoint and --num-classes."
            )
        predictor = Predictor.from_checkpoint(
            path=args.checkpoint,
            model_name=args.model,
            num_classes=args.num_classes,
            image_size=args.image_size,
            device=args.device,
        )

    # PIL.Image.open is lazy; .convert("RGB") forces 3 channels.
    image = Image.open(args.image).convert("RGB")
    probs, indices = predictor.predict(image, topk=args.topk)

    print(f"Top-{args.topk} predictions for {args.image} ({args.model}):")
    for rank, (prob, idx) in enumerate(zip(probs.tolist(), indices.tolist()), start=1):
        print(f"  {rank}. {predictor.class_name(idx):>16}  prob {prob:.4f}")


if __name__ == "__main__":
    main()
