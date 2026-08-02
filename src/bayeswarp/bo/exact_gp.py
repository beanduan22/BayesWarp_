from __future__ import annotations
import math
from typing import Optional, Tuple

import torch


class ExactGPSurrogate:
    def __init__(
        self,
        dim: int,
        device: torch.device,
        bound: float = 0.1,
        lengthscale: Optional[float] = None,
        outputscale: float = 1.0,
        noise: float = 0.01,
        lr: float = 0.05,
        jitter: float = 1e-6,
    ):
        self.dim = dim
        self.device = device
        self.bound = bound
        self.jitter = jitter
        self.dtype = torch.float64

        if lengthscale is None:
            lengthscale = max(bound, 1e-3)

        def _param(value: float) -> torch.Tensor:
            return torch.tensor(math.log(value), device=device, dtype=self.dtype, requires_grad=True)

        self.raw_lengthscale = _param(lengthscale)
        self.raw_outputscale = _param(outputscale)
        self.raw_noise = _param(noise)

        self.optimizer = torch.optim.Adam(
            [self.raw_lengthscale, self.raw_outputscale, self.raw_noise],
            lr=lr,
        )

        self.X: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None

    def _kernel(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        lengthscale = self.raw_lengthscale.exp()
        outputscale = self.raw_outputscale.exp()
        d2 = torch.cdist(A, B) ** 2
        return outputscale * torch.exp(-0.5 * d2 / (lengthscale ** 2 + 1e-12))

    def add_observation(self, u: torch.Tensor, y: float) -> None:
        u = u.detach().reshape(1, -1).to(device=self.device, dtype=self.dtype)
        y_t = torch.tensor([float(y)], device=self.device, dtype=self.dtype)
        self.X = u if self.X is None else torch.cat([self.X, u], dim=0)
        self.y = y_t if self.y is None else torch.cat([self.y, y_t], dim=0)

    def _chol(self) -> Tuple[torch.Tensor, torch.Tensor]:
        n = self.X.size(0)
        noise = self.raw_noise.exp()
        K = self._kernel(self.X, self.X)
        K = K + (noise + self.jitter) * torch.eye(n, device=self.device, dtype=self.dtype)
        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(self.y.unsqueeze(-1), L)
        return L, alpha

    def fit_step(self) -> Optional[float]:
        if self.X is None or self.X.size(0) == 0:
            return None
        self.optimizer.zero_grad(set_to_none=True)
        L, alpha = self._chol()
        n = self.X.size(0)
        log_ml = -0.5 * (self.y.unsqueeze(0) @ alpha).squeeze()
        log_ml = log_ml - torch.log(torch.diagonal(L)).sum()
        log_ml = log_ml - 0.5 * n * math.log(2.0 * math.pi)
        (-log_ml).backward()
        self.optimizer.step()
        return float(log_ml.detach().item())

    @torch.no_grad()
    def predict(self, U: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.X is None or self.X.size(0) == 0:
            n = U.size(0)
            return torch.zeros(n, device=self.device), torch.ones(n, device=self.device)
        U = U.to(device=self.device, dtype=self.dtype)
        L, alpha = self._chol()
        Kxs = self._kernel(U, self.X)
        mean = (Kxs @ alpha).squeeze(-1)
        v = torch.linalg.solve_triangular(L, Kxs.transpose(-1, -2), upper=False)
        var = self.raw_outputscale.exp() - (v ** 2).sum(dim=0)
        return mean.float(), var.clamp_min(1e-10).sqrt().float()
