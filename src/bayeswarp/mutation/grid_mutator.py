from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass
class GridMutator:
    """Grid parameterization of mutations restricted to the critical region.

    Mutations live on a coarse ``n x n`` grid per channel. A grid parameter
    vector ``u`` is mapped to pixel space by ``I_R``: bilinear upsampling to the
    image resolution followed by restriction to the critical region ``R``. Every
    input is reconstructed relative to the seed as ``x(u) = clip_p(x0 + I_R(u))``,
    so the accumulated mutation is never repeatedly added to the current input.
    """

    image_shape: Tuple[int, int, int]
    region_mask: torch.Tensor
    n: int
    r: float = 0.1
    eta: float = 0.1

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

    def clip_u(self, u: torch.Tensor) -> torch.Tensor:
        """Element-wise projection of grid parameters onto [-r, r]."""
        return u.clamp(-self.r, self.r)

    def sample_deltas(self, S: int, device: torch.device, delta_scale: float = 0.05) -> torch.Tensor:
        """Sample S candidate increments from a bounded uniform distribution."""
        return (torch.rand(S, self.dim, device=device) * 2.0 - 1.0) * delta_scale

    def interpolate_to_region(self, u: torch.Tensor) -> torch.Tensor:
        """I_R(u): bilinear upsampling followed by restriction to the region."""
        grid = u.view(1, self.c, self.n, self.n)
        up = F.interpolate(grid, size=(self.h, self.w), mode='bilinear', align_corners=False)
        return up.squeeze(0) * self.region_mask.to(u.device)

    def clip_p(self, x: torch.Tensor, x_seed: torch.Tensor) -> torch.Tensor:
        """Project pixels onto the relaxed original pixel range."""
        pmin = float(x_seed.min().item())
        pmax = float(x_seed.max().item())
        span = pmax - pmin
        return x.clamp(pmin - self.eta * span, pmax + self.eta * span)

    def reconstruct(self, u: torch.Tensor, x_seed: torch.Tensor) -> torch.Tensor:
        """x(u) = clip_p(x0 + I_R(u)), always relative to the seed."""
        return self.clip_p(x_seed + self.interpolate_to_region(u), x_seed)
