from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
import matplotlib.pyplot as plt

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run full CIFAR-10 experiments: baseline vs MixUp.")
	parser.add_argument("--epochs", type=int, default=200)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--data-dir", type=str, default="./data")
	parser.add_argument("--save-dir", type=str, default="./outputs")
	parser.add_argument("--batch-size", type=int, default=128)
	parser.add_argument("--num-workers", type=int, default=4)
	return parser.parse_args()


def run_command(command: list[str]) -> None:
	print("\nRunning:", " ".join(command))
	subprocess.run(command, check=True)


def plot_combined_curves(history_paths: dict[str, Path], save_path: Path):
	fig, axes = plt.subplots(1, 2, figsize=(14, 5))
	
	for name, path in history_paths.items():
		if not path.exists():
			continue
		with path.open("r", encoding="utf-8") as fp:
			history = json.load(fp)
		epochs = history.get("epoch", [])
		test_loss = history.get("test_loss", [])
		test_acc = history.get("test_acc", [])
		
		if not epochs:
			continue
		
		# Plot Test Loss
		axes[0].plot(epochs, test_loss, label=name)
		# Plot Test Acc
		axes[1].plot(epochs, test_acc, label=name)
		
	axes[0].set_title("Test Loss")
	axes[0].set_xlabel("Epoch")
	axes[0].set_ylabel("Cross-Entropy")
	axes[0].grid(True, alpha=0.3)
	axes[0].legend()
	
	axes[1].set_title("Test Accuracy")
	axes[1].set_xlabel("Epoch")
	axes[1].set_ylabel("Top-1 Acc (%)")
	axes[1].grid(True, alpha=0.3)
	axes[1].legend()

	fig.tight_layout()
	save_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(save_path, dpi=200)
	plt.close(fig)
	print(f"\nSaved combined curves to: {save_path}")


def summarize_results(summary_paths: dict[str, Path]):
	print("\n" + "="*40)
	print(f"{'Experiment':<20} | {'Test Accuracy (%)':<15}")
	print("="*40)
	
	for name, path in summary_paths.items():
		if not path.exists():
			print(f"{name:<20} | {'N/A (Failed/Missing)':<15}")
			continue
		with path.open("r", encoding="utf-8") as fp:
			summary = json.load(fp)
		acc = summary.get("best_test_acc", "N/A")
		if isinstance(acc, float):
			print(f"{name:<20} | {acc:.2f}")
		else:
			print(f"{name:<20} | {acc}")
	print("="*40 + "\n")


def main() -> None:
	args = parse_args()
	
	# Resolve the absolute path to the training script based on this file's location
	# This ensures it runs correctly even if the current working directory is different (like in Kaggle)
	ROOT = Path(__file__).resolve().parents[1]
	train_script = ROOT / "training" / "train.py"
	
	# The paper ablations for CIFAR
	experiments = [
		{"name": "baseline", "alpha": 0.0},
		{"name": "mixup_alpha_0.2", "alpha": 0.2},
		{"name": "mixup_alpha_0.4", "alpha": 0.4},
		{"name": "mixup_alpha_1.0", "alpha": 1.0},
	]

	history_paths = {}
	summary_paths = {}

	for exp in experiments:
		name = exp["name"]
		alpha = exp["alpha"]
		
		cmd = [
			sys.executable,
			str(train_script),
			"--dataset", "cifar10",
			"--epochs", str(args.epochs),
			"--seed", str(args.seed),
			"--data-dir", args.data_dir,
			"--save-dir", args.save_dir,
			"--run-name", name,
			"--batch-size", str(args.batch_size),
			"--num-workers", str(args.num_workers),
		]
		
		if alpha > 0:
			cmd.extend(["--mixup-alpha", str(alpha)])
			
		run_command(cmd)
		
		run_dir = Path(args.save_dir) / name
		history_paths[name] = run_dir / "history.json"
		summary_paths[name] = run_dir / "summary.json"

	# Generate ablation table and combined curves
	summarize_results(summary_paths)
	plot_combined_curves(history_paths, Path(args.save_dir) / "ablation_curves.png")

if __name__ == "__main__":
	main()
