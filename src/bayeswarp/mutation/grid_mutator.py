from __future__ import annotations
from typing import Tuple

import torch
import torch.nn.functional as F


class RegionMutator:
    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        region_mask: torch.Tensor,
        p_min: torch.Tensor,
        p_max: torch.Tensor,
        r: float = 0.1,
        eta: float = 0.1,
    ):
        self.c, self.h, self.w = image_shape
        mask = region_mask.float()
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        self.region_mask = mask
        self.r = float(r)
        self.eta = float(eta)
        device = mask.device
        lo = p_min.to(device).view(self.c, 1, 1)
        hi = p_max.to(device).view(self.c, 1, 1)
        span = hi - lo
        self.lo = lo - self.eta * span
        self.hi = hi + self.eta * span
        self.u_bound = float(r)

    @property
    def dim(self) -> int:
        raise NotImplementedError

    def interpolate_to_region(self, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def zero_params(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.dim, device=device)

    def clip_u(self, u: torch.Tensor) -> torch.Tensor:
        return u.clamp(-self.u_bound, self.u_bound)

    def sample_deltas(self, S: int, device: torch.device) -> torch.Tensor:
        return (torch.rand(S, self.dim, device=device) * 2.0 - 1.0) * self.r

    def clip_p(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.min(x, self.hi.to(x.device)), self.lo.to(x.device))

    def reconstruct(self, u: torch.Tensor, x_seed: torch.Tensor) -> torch.Tensor:
        return self.clip_p(x_seed + self.interpolate_to_region(u))


class GridMutator(RegionMutator):
    def __init__(self, image_shape, region_mask, p_min, p_max, n: int, r: float = 0.1, eta: float = 0.1):
        super().__init__(image_shape, region_mask, p_min, p_max, r=r, eta=eta)
        self.n = int(n)

    @property
    def dim(self) -> int:
        return self.n * self.n

    def interpolate_to_region(self, u: torch.Tensor) -> torch.Tensor:
        grid = u.view(1, 1, self.n, self.n)
        up = F.interpolate(grid, size=(self.h, self.w), mode='bilinear', align_corners=False)
        spatial = up.view(self.h, self.w) * self.region_mask.to(u.device)
        return spatial.unsqueeze(0).expand(self.c, self.h, self.w)


class PixelMutator(RegionMutator):
    def __init__(self, image_shape, region_mask, p_min, p_max, r: float = 0.1, eta: float = 0.1):
        super().__init__(image_shape, region_mask, p_min, p_max, r=r, eta=eta)
        self.indices = torch.nonzero(self.region_mask.reshape(-1) > 0, as_tuple=False).reshape(-1)

    @property
    def dim(self) -> int:
        return int(self.indices.numel())

    def interpolate_to_region(self, u: torch.Tensor) -> torch.Tensor:
        flat = torch.zeros(self.h * self.w, device=u.device, dtype=u.dtype)
        flat = flat.index_copy(0, self.indices.to(u.device), u)
        spatial = flat.view(self.h, self.w)
        return spatial.unsqueeze(0).expand(self.c, self.h, self.w)


def build_mutator(
    image_shape: Tuple[int, int, int],
    region_mask: torch.Tensor,
    p_min: torch.Tensor,
    p_max: torch.Tensor,
    n: int,
    r: float,
    eta: float,
    parameterization: str = 'grid',
) -> RegionMutator:
    if parameterization == 'grid':
        return GridMutator(image_shape, region_mask, p_min, p_max, n=n, r=r, eta=eta)
    if parameterization == 'pixel':
        return PixelMutator(image_shape, region_mask, p_min, p_max, r=r, eta=eta)
    raise ValueError(f'Unsupported parameterization: {parameterization}')
