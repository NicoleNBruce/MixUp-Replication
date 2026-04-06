from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run baseline and first MixUp CIFAR-10 experiments.")
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


def main() -> None:
	args = parse_args()
	train_script = Path("training") / "train.py"

	baseline_cmd = [
		sys.executable,
		str(train_script),
		"--dataset",
		"cifar10",
		"--epochs",
		str(args.epochs),
		"--seed",
		str(args.seed),
		"--data-dir",
		args.data_dir,
		"--save-dir",
		args.save_dir,
		"--run-name",
		"baseline",
		"--batch-size",
		str(args.batch_size),
		"--num-workers",
		str(args.num_workers),
	]

	mixup_cmd = [
		sys.executable,
		str(train_script),
		"--dataset",
		"cifar10",
		"--epochs",
		str(args.epochs),
		"--seed",
		str(args.seed),
		"--data-dir",
		args.data_dir,
		"--save-dir",
		args.save_dir,
		"--run-name",
		"mixup_alpha_1.0",
		"--mixup-alpha",
		"1.0",
		"--batch-size",
		str(args.batch_size),
		"--num-workers",
		str(args.num_workers),
	]

	run_command(baseline_cmd)
	run_command(mixup_cmd)


if __name__ == "__main__":
	main()
