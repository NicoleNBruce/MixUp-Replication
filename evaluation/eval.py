from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from data.dataloader import get_cifar_loaders
from models.resnet import resnet18


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on CIFAR.")
	parser.add_argument("--checkpoint", type=str, required=True)
	parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
	parser.add_argument("--data-dir", type=str, default="./data")
	parser.add_argument("--batch-size", type=int, default=256)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--history", type=str, default="")
	parser.add_argument("--plot-path", type=str, default="")
	return parser.parse_args()


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device: torch.device):
	model.eval()
	total_loss = 0.0
	total_correct = 0
	total_samples = 0

	for images, targets in loader:
		images = images.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)
		logits = model(images)
		loss = criterion(logits, targets)

		batch_size = targets.size(0)
		total_loss += loss.item() * batch_size
		total_correct += logits.argmax(dim=1).eq(targets).sum().item()
		total_samples += batch_size

	return total_loss / total_samples, 100.0 * total_correct / total_samples


def plot_history(history_path: Path, plot_path: Path) -> None:
	with history_path.open("r", encoding="utf-8") as fp:
		history = json.load(fp)

	epochs = history.get("epoch", [])
	train_loss = history.get("train_loss", [])
	test_loss = history.get("test_loss", [])
	train_acc = history.get("train_acc", [])
	test_acc = history.get("test_acc", [])

	if not epochs:
		print("history.json has no epoch data. Skipping plot.")
		return

	fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

	axes[0].plot(epochs, train_loss, label="train")
	axes[0].plot(epochs, test_loss, label="test")
	axes[0].set_title("Loss")
	axes[0].set_xlabel("Epoch")
	axes[0].set_ylabel("Cross-Entropy")
	axes[0].grid(True, alpha=0.2)
	axes[0].legend()

	axes[1].plot(epochs, train_acc, label="train")
	axes[1].plot(epochs, test_acc, label="test")
	axes[1].set_title("Accuracy")
	axes[1].set_xlabel("Epoch")
	axes[1].set_ylabel("Top-1 (%)")
	axes[1].grid(True, alpha=0.2)
	axes[1].legend()

	fig.tight_layout()
	plot_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(plot_path, dpi=200)
	plt.close(fig)
	print(f"Saved plots to: {plot_path}")


def main() -> None:
	args = parse_args()
	checkpoint_path = Path(args.checkpoint)
	if not checkpoint_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	_, test_loader, num_classes = get_cifar_loaders(
		dataset=args.dataset,
		data_dir=args.data_dir,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		seed=42,
	)

	model = resnet18(num_classes=num_classes).to(device)
	payload = torch.load(checkpoint_path, map_location=device)
	model_state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
	model.load_state_dict(model_state)

	criterion = nn.CrossEntropyLoss()
	test_loss, test_acc = evaluate(model, test_loader, criterion, device)
	print(f"Checkpoint: {checkpoint_path}")
	print(f"Test loss: {test_loss:.4f}")
	print(f"Test accuracy: {test_acc:.2f}%")

	history_path = Path(args.history) if args.history else checkpoint_path.with_name("history.json")
	if history_path.exists():
		plot_path = Path(args.plot_path) if args.plot_path else checkpoint_path.with_name("curves.png")
		plot_history(history_path, plot_path)
	else:
		print(f"No history file found at {history_path}, skipping plots.")


if __name__ == "__main__":
	main()
