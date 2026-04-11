from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from data.cifar_c import CORRUPTIONS, get_cifar10c_loader
from models.resnet import resnet18


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on CIFAR-10-C (Hendrycks & Dietterich, 2019).")
	parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
	parser.add_argument("--data-dir", type=str, default="./data", help="Directory where CIFAR-10-C will be downloaded/stored")
	parser.add_argument("--batch-size", type=int, default=256)
	parser.add_argument("--num-workers", type=int, default=4)
	return parser.parse_args()


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader, device: torch.device):
	model.eval()
	total_correct = 0
	total_samples = 0

	for images, targets in loader:
		images = images.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)
		logits = model(images)

		batch_size = targets.size(0)
		total_correct += logits.argmax(dim=1).eq(targets).sum().item()
		total_samples += batch_size

	return 100.0 * total_correct / total_samples


def main() -> None:
	args = parse_args()
	
	checkpoint_path = Path(args.checkpoint)
	if not checkpoint_path.exists():
		print(f"Error: Checkpoint {checkpoint_path} not found.")
		return

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = resnet18(num_classes=10).to(device)
	
	payload = torch.load(checkpoint_path, map_location=device)
	model_state = payload.get("model_state", payload)
	model.load_state_dict(model_state)
	
	results: Dict[str, float] = {}
	
	print(f"\nEvaluating Model Checkpoint: {checkpoint_path.name}")
	print("-" * 50)
	print(f"{'Corruption Type':<25} | {'Accuracy (%)':<15}")
	print("-" * 50)
	
	avg_acc = 0.0
	for corruption in CORRUPTIONS:
		loader = get_cifar10c_loader(
			corruption=corruption,
			data_dir=args.data_dir,
			batch_size=args.batch_size,
			num_workers=args.num_workers
		)
		acc = evaluate_loader(model, loader, device)
		results[corruption] = acc
		avg_acc += acc
		print(f"{corruption:<25} | {acc:.2f}%")
		
	avg_acc /= len(CORRUPTIONS)
	print("-" * 50)
	print(f"{'Mean Accuracy':<25} | {avg_acc:.2f}%")
	print("=" * 50)


if __name__ == "__main__":
	main()
