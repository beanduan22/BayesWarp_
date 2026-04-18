from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import torch


@dataclass
class SVGPSurrogate:
    dim: int
    m: int
    device: torch.device
    lengthscale: float = 0.5
    prior_precision: float = 1.0
    noise_precision: float = 25.0

    def __post_init__(self):
        self.Z = (torch.rand(self.m, self.dim, device=self.device) * 2.0) - 1.0
        self.X = None
        self.y = None
        self.w_mean = torch.zeros(self.m, device=self.device)
        self.w_cov = torch.eye(self.m, device=self.device) / self.prior_precision

    def _phi(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim == 1:
            X = X.unsqueeze(0)
        d2 = torch.cdist(X, self.Z) ** 2
        return torch.exp(-0.5 * d2 / (self.lengthscale ** 2 + 1e-12))

    def add_observation(self, x: torch.Tensor, y: torch.Tensor):
        x = x.detach().view(1, -1).to(self.device)
        y = y.detach().view(-1).to(self.device)
        self.X = x if self.X is None else torch.cat([self.X, x], dim=0)
        self.y = y if self.y is None else torch.cat([self.y, y], dim=0)

    def fit_step(self, iters: int = 1):
        if self.X is None or self.X.size(0) == 0:
            return
        Phi = self._phi(self.X)
        A = self.prior_precision * torch.eye(self.m, device=self.device) + self.noise_precision * (Phi.T @ Phi)
        b = self.noise_precision * (Phi.T @ self.y)
        self.w_cov = torch.linalg.pinv(A)
        self.w_mean = self.w_cov @ b

    @torch.no_grad()
    def predict(self, Xcand: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Phi = self._phi(Xcand.to(self.device))
        mu = Phi @ self.w_mean
        cov = Phi @ self.w_cov
        var = (cov * Phi).sum(dim=1) + (1.0 / self.noise_precision)
        return mu, var.clamp_min(1e-8).sqrt()
