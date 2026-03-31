from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn.functional as F


@dataclass
class GridMutator:
    image_shape: Tuple[int, int, int]
    region_mask: torch.Tensor
    n: int
    clip_min: float = -0.1
    clip_max: float = 0.1

    def __post_init__(self):
        c, h, w = self.image_shape
        self.c = c
        self.h = h
        self.w = w
        self.dim = self.c * self.n * self.n
        if self.region_mask.ndim == 2:
            self.region_mask = self.region_mask.unsqueeze(0)
        self.region_mask = self.region_mask.float()

    def zero_params(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.dim, device=device)

    def sample_candidate_deltas(self, current_u: torch.Tensor, S: int, delta_scale: float = 0.05) -> torch.Tensor:
        candidates = []
        for _ in range(S):
            delta = (torch.rand_like(current_u) * 2 - 1) * delta_scale
            candidates.append((current_u + delta).clamp(self.clip_min, self.clip_max))
        return torch.stack(candidates, dim=0)

    def upsample(self, u: torch.Tensor) -> torch.Tensor:
        grid = u.view(1, self.c, self.n, self.n)
        up = F.interpolate(grid, size=(self.h, self.w), mode='bilinear', align_corners=False)
        return up.squeeze(0) * self.region_mask.to(u.device)

    def apply(self, x: torch.Tensor, u: torch.Tensor, eta: float = 0.1) -> torch.Tensor:
        delta_x = self.upsample(u)
        x_new = x + delta_x
        pmin = float(x.min().item())
        pmax = float(x.max().item())
        lo = pmin - eta * (pmax - pmin)
        hi = pmax + eta * (pmax - pmin)
        return x_new.clamp(lo, hi)
