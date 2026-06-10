"""
Forward-pass latency benchmark across all registered models.

This script measures single-forward-pass latency for every model in the unified
registry (or a filtered subset), so the merged library's variants -- MobileViT,
EfficientViT, EfficientFormer, and the direct torchvision baselines -- can be
compared on identical, fair terms.

What it measures, per model
---------------------------
1. Build the model (random weights), move to the device, switch to eval mode.
2. Run ``warmup`` untimed forward passes (settle CUDA kernels / caches).
3. Run ``iters`` timed forward passes under ``torch.no_grad`` (no backward).
4. Report parameter count and latency statistics: mean +/- std, median, and p95
   in milliseconds.

On CUDA, timing uses ``torch.cuda.Event`` with synchronization; on CPU it uses
``time.perf_counter``. All models are built at the same ``--size`` (224 by
default, which satisfies EfficientViT's /16 and EfficientFormer's /32 size
constraints); a model that cannot run at the requested size is skipped with a
short reason rather than aborting the whole run.

Example
-------
python scripts/latency.py --size 224 --batch-size 1 --iters 100
python scripts/latency.py --family efficientvit --iters 50
python scripts/latency.py --models mobilevit_xs torchvision_mobilenet_v2
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

# Make the package root importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch import nn

from models import count_parameters_millions, create_model, list_families, list_models


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the benchmark."""
    parser = argparse.ArgumentParser(
        description="Latency benchmark across registered models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--size", type=int, default=224, help="Input spatial size for all models.")
    parser.add_argument("--batch-size", type=int, default=1, help="Input batch size.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations.")
    parser.add_argument("--device", default=None, help="cuda | cpu (default: auto).")
    parser.add_argument("--family", default=None, choices=list_families(), help="Restrict to one family.")
    parser.add_argument("--models", nargs="+", default=None, help="Explicit list of model names.")
    return parser.parse_args()


def resolve_device(device: str | None) -> torch.device:
    """Pick the user-requested device, or CUDA if available, otherwise CPU."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    x: torch.Tensor,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, float]:
    """
    Time the forward pass and return latency statistics in milliseconds.

    The first ``warmup`` calls are untimed; the next ``iters`` are individually
    timed. Returns a dict with ``mean``, ``std``, ``median`` and ``p95``.
    """
    model = model.to(device).eval()
    x = x.to(device)

    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    timings: list[float] = []
    if device.type == "cuda":
        # CUDA events record GPU-side timestamps; elapsed_time returns ms.
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(x)
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))
    else:
        # CPU: perf_counter returns seconds; convert to milliseconds.
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            timings.append((time.perf_counter() - t0) * 1000.0)

    timings.sort()
    # p95 = the 95th-percentile latency (index ceil(0.95 * n) - 1).
    p95_index = max(0, min(len(timings) - 1, int(round(0.95 * len(timings))) - 1))
    return {
        "mean": statistics.mean(timings),
        "std": statistics.stdev(timings) if len(timings) > 1 else 0.0,
        "median": statistics.median(timings),
        "p95": timings[p95_index],
    }


def select_models(args: argparse.Namespace) -> list[str]:
    """Resolve which model names to benchmark from the CLI filters."""
    if args.models:
        return args.models
    if args.family:
        return list_models(family=args.family)
    return list_models()


def main() -> None:
    """Build each selected model, benchmark it, and print a comparison table."""
    args = parse_args()
    device = resolve_device(args.device)
    names = select_models(args)

    print(f"Device: {device}")
    print(f"Input shape: ({args.batch_size}, 3, {args.size}, {args.size})")
    print(f"Warmup iters: {args.warmup} | Timed iters: {args.iters}")
    print(f"Models: {len(names)}")
    print()
    header = f"{'Model':<34} {'Params(M)':>10} {'mean+/-std (ms)':>18} {'median':>9} {'p95':>9}"
    print(header)
    print("-" * len(header))

    x = torch.randn(args.batch_size, 3, args.size, args.size)

    for name in names:
        try:
            # num_classes=1000 mirrors ImageNet-scale heads; image_size=size
            # builds fixed-resolution models at the benchmarked resolution.
            model = create_model(name, num_classes=1000, image_size=args.size)
            params_m = count_parameters_millions(model)
            stats = measure_latency(model, x, args.warmup, args.iters, device)
            mean_std = f"{stats['mean']:.2f} +/- {stats['std']:.2f}"
            print(
                f"{name:<34} {params_m:>10.2f} {mean_std:>18} "
                f"{stats['median']:>9.2f} {stats['p95']:>9.2f}"
            )
        except Exception as exc:  # noqa: BLE001 - report and continue
            # E.g. a fixed-resolution model that rejects --size; keep going.
            reason = str(exc).splitlines()[0][:48]
            print(f"{name:<34} {'-':>10} {'skipped: ' + reason:>40}")


if __name__ == "__main__":
    main()
