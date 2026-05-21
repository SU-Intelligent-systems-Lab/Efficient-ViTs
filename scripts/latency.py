"""
Forward-pass latency benchmark for MobileViT and MobileNetV2.

This script compares the inference latency of the three MobileViT variants
against torch-vision's MobileNetV2 on random input tensors. Models use random
initialization. No actual images, no trained weights, no backward pass.

Example
-------
python scripts/latency.py --size 256 --batch-size 1 --iters 100
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch import nn
from torchvision.models import mobilenet_v2

from models.mobilevit import mobilevit_s, mobilevit_xs, mobilevit_xxs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latency benchmark.")
    parser.add_argument("--size", type=int, default=256, help="Input spatial size.")
    parser.add_argument("--batch-size", type=int, default=1, help="Input batch size.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=100, help="Timed iterations.")
    parser.add_argument("--device", default=None, help="Device, e.g. 'cuda' or 'cpu'. Default: auto.")
    return parser.parse_args()


def resolve_device(device: str | None) -> torch.device:
    """Pick the user-requested device, or CUDA if available, otherwise CPU."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """Return the total number of model parameters."""
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    x: torch.Tensor,
    warmup: int,
    iters: int,
    device: torch.device,
) -> tuple[float, float]:
    """
    Return mean and standard deviation of forward-pass latency in milliseconds.

    On CUDA the timing uses torch.cuda.Event for accuracy. On CPU it uses
    time.perf_counter.
    """
    model = model.to(device).eval()
    x = x.to(device)

    for _ in range(warmup):
        model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    timings: list[float] = []

    if device.type == "cuda":
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(x)
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)

    mean = statistics.mean(timings)
    std = statistics.stdev(timings) if len(timings) > 1 else 0.0
    return mean, std


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    builders: dict[str, Callable[[], nn.Module]] = {
        "MobileViT-XXS": lambda: mobilevit_xxs(num_classes=1000),
        "MobileViT-XS":  lambda: mobilevit_xs(num_classes=1000),
        "MobileViT-S":   lambda: mobilevit_s(num_classes=1000),
        "MobileNetV2":   lambda: mobilenet_v2(weights=None),
    }

    x = torch.randn(args.batch_size, 3, args.size, args.size)

    print(f"Device: {device}")
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Warmup iters: {args.warmup} | Timed iters: {args.iters}")
    print()
    print(f"{'Model':<16} {'Params (M)':>12} {'Latency (ms)':>22}")
    print("-" * 52)
    for name, build in builders.items():
        model = build()
        params_m = count_parameters(model) / 1e6
        mean, std = measure_latency(
            model=model,
            x=x,
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )
        latency_str = f"{mean:6.2f} +/- {std:5.2f}"
        print(f"{name:<16} {params_m:>12.2f} {latency_str:>22}")


if __name__ == "__main__":
    main()
