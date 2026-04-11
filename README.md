# MixUp Replication (CIFAR-10)

This repository is set up to replicate the MixUp paper pipeline in clear phases:

1. Train a baseline ResNet-18 on CIFAR-10 without MixUp.
2. Train the same model with MixUp enabled.
3. Keep runs reproducible with fixed seeds.
4. Evaluate checkpoints and plot loss/accuracy curves.

The code is runnable end-to-end today, and is intentionally minimal.

## Project Layout

```text
mixup-replication/
├── README.md
├── requirements.txt
├── data/
│   └── dataloader.py
├── models/
│   └── resnet.py
├── training/
│   ├── train.py
│   └── mixup.py
├── evaluation/
│   └── eval.py
└── experiments/
		└── run_experiments.py
```

## Setup

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Phase 1: Baseline (No MixUp)

Train standard CIFAR ResNet-18 baseline:

```bash
python training/train.py \
	--dataset cifar10 \
	--epochs 200 \
	--batch-size 128 \
	--lr 0.1 \
	--weight-decay 5e-4 \
	--seed 42 \
	--run-name baseline
```

Expected: around ~93% test accuracy with this common setup (exact value can vary by hardware/PyTorch/CUDA versions).

Artifacts are saved under:

```text
outputs/baseline/
	best.pt
	last.pt
	history.json
	summary.json
```

## Phase 2: MixUp Module + Training

MixUp is implemented in `training/mixup.py` as a standalone module (compact helper functions).

Run first MixUp experiment on CIFAR-10:

```bash
python training/train.py \
	--dataset cifar10 \
	--epochs 200 \
	--batch-size 128 \
	--lr 0.1 \
	--weight-decay 5e-4 \
	--seed 42 \
	--mixup-alpha 1.0 \
	--run-name mixup_alpha_1.0
```

Artifacts are saved under `outputs/mixup_alpha_1.0/`.

## Reproducibility

Training uses fixed seeds and deterministic CuDNN mode.

To verify reproducibility, rerun with same arguments and seed:

```bash
python training/train.py --dataset cifar10 --epochs 200 --seed 42 --run-name baseline_repro
```

Compare `summary.json` and `history.json` between runs.

## One-Command Run for Baseline + MixUp Ablations

```bash
python experiments/run_experiments.py --epochs 200 --seed 42 --dataset cifar10
```

This executes:
1. Baseline training (`run-name=cifar10_baseline`)
2. MixUp training with α=0.2, 0.4, 1.0 (`run-name=cifar10_mixup_alpha_...`)
3. Prints a comparison table and plots `ablation_curves.png`.

To run on **CIFAR-100**, just swap the dataset flag:
```bash
python experiments/run_experiments.py --epochs 200 --seed 42 --dataset cifar100
```

## Evaluation and Curves

Evaluate a checkpoint and generate curves from history:

```bash
python evaluation/eval.py --checkpoint outputs/baseline/best.pt
python evaluation/eval.py --checkpoint outputs/mixup_alpha_1.0/best.pt
```

By default, the script looks for `history.json` next to the checkpoint and writes `curves.png` in the same folder.

## Evaluate on CIFAR-10-C (Domain Shift)

You can test trained checkpoints against **CIFAR-10-C** (corrupted images by Hendrycks et al.) to see if MixUp genuinely helps with robustness against domain shifts in a way standard training doesn't. 

This will automatically download and extract CIFAR-10-C (about 2.5 GB) into `--data-dir`, and output the average accuracy over all 15 corruption types:
```bash
python evaluation/eval_cifar_c.py \
  --checkpoint outputs/mixup_alpha_1.0/best.pt \
  --data-dir ./data
```

## Suggested Commit Flow

Phase commit (baseline):

```bash
git add data/ models/ training/train.py requirements.txt README.md
git commit -m "Add CIFAR-10 baseline training with ResNet-18"
```

Phase commit (MixUp + experiments + eval):

```bash
git add training/mixup.py evaluation/ experiments/ README.md
git commit -m "Add MixUp module, experiments runner, reproducibility and eval tooling"
```

## Notes

- No final paper-quality results are required yet; this repo is focused on a working reproducible pipeline.
- If you want PreActResNet-18 next, it can be added as an alternate model flag in the same training script.
