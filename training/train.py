from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from data.dataloader import get_cifar_loaders
from models.resnet import resnet18
from training.mixup import mixup_criterion, mixup_data


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train ResNet-18 baseline or MixUp on CIFAR.")
	parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
	parser.add_argument("--data-dir", type=str, default="./data")
	parser.add_argument("--epochs", type=int, default=200)
	parser.add_argument("--batch-size", type=int, default=128)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--lr", type=float, default=0.1)
	parser.add_argument("--momentum", type=float, default=0.9)
	parser.add_argument("--weight-decay", type=float, default=5e-4)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--mixup-alpha", type=float, default=0.0)
	parser.add_argument("--save-dir", type=str, default="./outputs")
	parser.add_argument("--run-name", type=str, default="baseline")
	return parser.parse_args()


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	cudnn.deterministic = True
	cudnn.benchmark = False


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
	preds = logits.argmax(dim=1)
	return (preds.eq(targets).sum().item() / targets.size(0)) * 100.0


def train_one_epoch(
	model: nn.Module,
	loader,
	criterion,
	optimizer,
	device: torch.device,
	mixup_alpha: float,
) -> Tuple[float, float]:
	model.train()
	running_loss = 0.0
	running_correct = 0.0
	total = 0

	progress = tqdm(loader, desc="train", leave=False)
	for images, targets in progress:
		images = images.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)

		optimizer.zero_grad(set_to_none=True)

		if mixup_alpha > 0:
			mixed_x, y_a, y_b, lam = mixup_data(images, targets, mixup_alpha, device)
			logits = model(mixed_x)
			loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
			preds = logits.argmax(dim=1)
			correct = lam * preds.eq(y_a).sum().item() + (1.0 - lam) * preds.eq(y_b).sum().item()
		else:
			logits = model(images)
			loss = criterion(logits, targets)
			correct = logits.argmax(dim=1).eq(targets).sum().item()

		loss.backward()
		optimizer.step()

		batch_size = targets.size(0)
		running_loss += loss.item() * batch_size
		running_correct += correct
		total += batch_size

		progress.set_postfix(loss=running_loss / total, acc=100.0 * running_correct / total)

	return running_loss / total, 100.0 * running_correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device: torch.device) -> Tuple[float, float]:
	model.eval()
	running_loss = 0.0
	running_correct = 0
	total = 0

	progress = tqdm(loader, desc="eval ", leave=False)
	for images, targets in progress:
		images = images.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)

		logits = model(images)
		loss = criterion(logits, targets)

		batch_size = targets.size(0)
		running_loss += loss.item() * batch_size
		running_correct += logits.argmax(dim=1).eq(targets).sum().item()
		total += batch_size

	return running_loss / total, 100.0 * running_correct / total


def save_checkpoint(path: Path, model: nn.Module, optimizer, scheduler, epoch: int, best_acc: float, args: argparse.Namespace):
	payload = {
		"epoch": epoch,
		"best_acc": best_acc,
		"model_state": model.state_dict(),
		"optimizer_state": optimizer.state_dict(),
		"scheduler_state": scheduler.state_dict(),
		"args": vars(args),
	}
	torch.save(payload, path)


def main() -> None:
	args = parse_args()
	set_seed(args.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	train_loader, test_loader, num_classes = get_cifar_loaders(
		dataset=args.dataset,
		data_dir=args.data_dir,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		seed=args.seed,
	)

	model = resnet18(num_classes=num_classes).to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(
		model.parameters(),
		lr=args.lr,
		momentum=args.momentum,
		weight_decay=args.weight_decay,
		nesterov=True,
	)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

	run_dir = Path(args.save_dir) / args.run_name
	run_dir.mkdir(parents=True, exist_ok=True)

	history: Dict[str, List[float]] = {
		"epoch": [],
		"train_loss": [],
		"train_acc": [],
		"test_loss": [],
		"test_acc": [],
		"lr": [],
	}

	best_acc = 0.0
	for epoch in range(1, args.epochs + 1):
		train_loss, train_acc = train_one_epoch(
			model=model,
			loader=train_loader,
			criterion=criterion,
			optimizer=optimizer,
			device=device,
			mixup_alpha=args.mixup_alpha,
		)
		test_loss, test_acc = evaluate(model, test_loader, criterion, device)
		scheduler.step()

		history["epoch"].append(epoch)
		history["train_loss"].append(train_loss)
		history["train_acc"].append(train_acc)
		history["test_loss"].append(test_loss)
		history["test_acc"].append(test_acc)
		history["lr"].append(optimizer.param_groups[0]["lr"])

		if test_acc > best_acc:
			best_acc = test_acc
			save_checkpoint(run_dir / "best.pt", model, optimizer, scheduler, epoch, best_acc, args)

		save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, epoch, best_acc, args)

		print(
			f"Epoch {epoch:03d}/{args.epochs} | "
			f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
			f"test_loss={test_loss:.4f} test_acc={test_acc:.2f}% | "
			f"best={best_acc:.2f}%"
		)

	with (run_dir / "history.json").open("w", encoding="utf-8") as fp:
		json.dump(history, fp, indent=2)

	with (run_dir / "summary.json").open("w", encoding="utf-8") as fp:
		json.dump(
			{
				"best_test_acc": best_acc,
				"final_test_acc": history["test_acc"][-1],
				"mixup_alpha": args.mixup_alpha,
				"seed": args.seed,
			},
			fp,
			indent=2,
		)

	print(f"Training completed. Best test accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
	main()
