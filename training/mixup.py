from __future__ import annotations

import numpy as np
import torch


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float, device: torch.device):
	if alpha <= 0:
		return x, y, y, 1.0
	lam = np.random.beta(alpha, alpha)
	batch_size = x.size(0)
	index = torch.randperm(batch_size, device=device)
	mixed_x = lam * x + (1 - lam) * x[index]
	y_a, y_b = y, y[index]
	return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float):
	return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
