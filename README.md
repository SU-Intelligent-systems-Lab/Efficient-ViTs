# MobileViT on CIFAR-100

PyTorch implementation of MobileViT (XXS / XS / S) trained on CIFAR-100, with a
torchvision MobileNetV2 baseline for comparison and a latency benchmark.

## Project layout

```
MobileViT/
├── data/                 CIFAR-100 dataset + train/eval transforms + loaders
├── models/               MobileViT blocks, configs, and the assembled networks
│                         (also a local MobileNetV2 reference impl)
├── training/             Trainer, optimizer/scheduler builders, checkpoint I/O
├── inference/            Predictor used by infer.py
└── scripts/
    ├── train.py                 Train MobileViT-XS on CIFAR-100
    ├── train_mobilenetv2.py     Train torchvision MobileNetV2 on CIFAR-100
    ├── infer.py                 Top-k inference on a single image
    └── latency.py               Forward-pass latency benchmark
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision numpy pillow
```

CUDA is auto-detected; pass `--device cpu` to force CPU.

## Training

### MobileViT-XS

```powershell
python scripts/train.py --output mobilevit_xs_cifar100.pt --epochs 100
```

Defaults: image size 32, batch 128, AdamW (lr 1e-3, wd 0.05), cosine schedule,
early-stopping patience 10. Logs to `train.log`.

### MobileNetV2 (torchvision) baseline

```powershell
python scripts/train_mobilenetv2.py --output mobilenetv2_cifar100.pt
```

Same data pipeline and trainer; the script uses
`torchvision.models.mobilenet_v2(weights=None, num_classes=100)`. Defaults
are tuned for MobileNetV2's heavier downsampling (image size 128, batch 32,
30 epochs). Logs to `train_mobilenetv2.log`.

CIFAR-100 is downloaded automatically into `--data-root` (default `./cifar100`)
on first run.

## Inference

```powershell
python scripts/infer.py --checkpoint mobilevit_xs_cifar100.pt --image path\to\img.jpg
```

## Latency benchmark

```powershell
python scripts/latency.py --size 256 --batch-size 1 --iters 100
```

Compares forward-pass latency of MobileViT-XXS / XS / S against torchvision
MobileNetV2 on randomly initialized weights.

## Notes

- Both training scripts share `data/`, `training/`, and the same CLI surface
  (epochs, image size, batch size, lr, weight decay, early-stopping patience,
  num-workers, seed, device).
- Each script writes a separate log file so MobileViT and MobileNetV2 runs do
  not overwrite each other.
- Checkpoints are plain `state_dict` files saved via `training.save_model`.
