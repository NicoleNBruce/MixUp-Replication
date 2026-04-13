from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR_STATS = {
	"cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	"cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


def _seed_worker(worker_id: int) -> None:
	worker_seed = torch.initial_seed() % 2**32
	np.random.seed(worker_seed)
	random.seed(worker_seed)


def get_cifar_loaders(
	dataset: str = "cifar10",
	data_dir: str = "./data",
	batch_size: int = 128,
	num_workers: int = 4,
	seed: int = 42,
	corrupt_prob: float = 0.0,
) -> Tuple[DataLoader, DataLoader, int]:
	dataset = dataset.lower()
	if dataset not in CIFAR_STATS:
		raise ValueError(f"Unsupported dataset: {dataset}. Use 'cifar10' or 'cifar100'.")

	mean, std = CIFAR_STATS[dataset]

	train_transform = transforms.Compose(
		[
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean, std),
		]
	)
	test_transform = transforms.Compose(
		[
			transforms.ToTensor(),
			transforms.Normalize(mean, std),
		]
	)

	dataset_cls = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100

	train_set = dataset_cls(root=data_dir, train=True, transform=train_transform, download=True)
	test_set = dataset_cls(root=data_dir, train=False, transform=test_transform, download=True)
	num_classes = 10 if dataset == "cifar10" else 100

	if corrupt_prob > 0.0:
		print(f"Corrupting {corrupt_prob*100}% of training labels...")
		random.seed(seed)
		num_to_corrupt = int(corrupt_prob * len(train_set))
		indices = list(range(len(train_set)))
		random.shuffle(indices)
		for i in range(num_to_corrupt):
			train_set.targets[indices[i]] = random.randint(0, num_classes - 1)

	generator = torch.Generator()
	generator.manual_seed(seed)

	train_loader = DataLoader(
		train_set,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=True,
		worker_init_fn=_seed_worker,
		generator=generator,
	)
	test_loader = DataLoader(
		test_set,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True,
	)
	return train_loader, test_loader, num_classes
