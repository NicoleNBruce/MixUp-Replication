from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from data.dataloader import CIFAR_STATS

CIFAR10C_URL = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"

CORRUPTIONS = [
	"gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
	"glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
	"brightness", "contrast", "elastic_transform", "pixelate",
	"jpeg_compression"
]

class CIFAR10C(Dataset):
	def __init__(self, root: str, corruption: str, transform=None):
		self.root = Path(root) / "CIFAR-10-C"
		self.corruption = corruption
		self.transform = transform
		self._ensure_downloaded()

		# Load exactly the corruption array and the labels
		# The arrays in CIFAR-10-C usually have shape (50000, 32, 32, 3) (10k images x 5 severities)
		images_path = self.root / f"{corruption}.npy"
		labels_path = self.root / "labels.npy"

		self.images = np.load(images_path)
		self.labels = np.load(labels_path)

	def _ensure_downloaded(self):
		if not self.root.exists():
			tar_path = self.root.parent / "CIFAR-10-C.tar"
			if not tar_path.exists():
				print(f"Downloading CIFAR-10-C from {CIFAR10C_URL} ... this might take a while.")
				self.root.parent.mkdir(parents=True, exist_ok=True)
				urllib.request.urlretrieve(CIFAR10C_URL, tar_path)
			
			print("Extracting CIFAR-10-C ...")
			with tarfile.open(tar_path, "r") as tar:
				tar.extractall(path=self.root.parent)

	def __len__(self) -> int:
		return len(self.labels)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
		img = self.images[idx]
		label = self.labels[idx]

		if self.transform is not None:
			img = self.transform(img)

		return img, int(label)

def get_cifar10c_loader(
	corruption: str,
	data_dir: str = "./data",
	batch_size: int = 128,
	num_workers: int = 4
) -> DataLoader:
	mean, std = CIFAR_STATS["cifar10"]
	
	test_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean, std),
	])

	dataset = CIFAR10C(root=data_dir, corruption=corruption, transform=test_transform)
	
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True,
	)
