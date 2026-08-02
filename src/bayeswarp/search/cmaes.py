from __future__ import annotations
import math
from typing import List

import torch


class CMAES:
    def __init__(self, dim: int, device: torch.device, bound: float, sigma0: float, population: int):
        self.dim = int(dim)
        self.device = device
        self.bound = float(bound)
        self.sigma = float(sigma0)
        self.population = int(population)

        self.mean = torch.zeros(self.dim, device=device, dtype=torch.float64)
        self.C = torch.eye(self.dim, device=device, dtype=torch.float64)
        self.p_sigma = torch.zeros(self.dim, device=device, dtype=torch.float64)
        self.p_c = torch.zeros(self.dim, device=device, dtype=torch.float64)
        self.generation = 0

        self.mu = self.population // 2
        weights = torch.tensor(
            [math.log(self.mu + 0.5) - math.log(i + 1.0) for i in range(self.mu)],
            device=device,
            dtype=torch.float64,
        )
        self.weights = weights / weights.sum()
        self.mu_eff = float(1.0 / (self.weights ** 2).sum().item())

        n = float(self.dim)
        self.c_sigma = (self.mu_eff + 2.0) / (n + self.mu_eff + 5.0)
        self.d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((self.mu_eff - 1.0) / (n + 1.0)) - 1.0) + self.c_sigma
        self.c_c = (4.0 + self.mu_eff / n) / (n + 4.0 + 2.0 * self.mu_eff / n)
        self.c_1 = 2.0 / ((n + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
            1.0 - self.c_1,
            2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((n + 2.0) ** 2 + self.mu_eff),
        )
        self.chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

    def ask(self) -> List[torch.Tensor]:
        eigenvalues, eigenvectors = torch.linalg.eigh(self.C)
        eigenvalues = eigenvalues.clamp_min(1e-20)
        self._B = eigenvectors
        self._D = eigenvalues.sqrt()
        self._z = torch.randn(self.population, self.dim, device=self.device, dtype=torch.float64)
        offspring = self.mean.unsqueeze(0) + self.sigma * (self._z * self._D.unsqueeze(0)) @ self._B.transpose(0, 1)
        offspring = offspring.clamp(-self.bound, self.bound)
        return [offspring[i].float() for i in range(self.population)]

    def tell(self, solutions: List[torch.Tensor], values: List[float]) -> None:
        if not solutions:
            return
        order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
        selected = torch.stack([solutions[i].to(dtype=torch.float64) for i in order[: self.mu]], dim=0)
        weights = self.weights[: selected.size(0)]
        weights = weights / weights.sum()

        old_mean = self.mean.clone()
        self.mean = (weights.unsqueeze(1) * selected).sum(dim=0)

        inv_sqrt_C = self._B @ torch.diag(1.0 / self._D) @ self._B.transpose(0, 1)
        delta_mean = (self.mean - old_mean) / max(self.sigma, 1e-12)
        self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + math.sqrt(
            self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff
        ) * (inv_sqrt_C @ delta_mean)

        self.generation += 1
        norm_p_sigma = float(self.p_sigma.norm().item())
        denom = math.sqrt(1.0 - (1.0 - self.c_sigma) ** (2 * self.generation))
        h_sigma = 1.0 if norm_p_sigma / max(denom, 1e-12) < (1.4 + 2.0 / (self.dim + 1.0)) * self.chi_n else 0.0

        self.p_c = (1.0 - self.c_c) * self.p_c + h_sigma * math.sqrt(
            self.c_c * (2.0 - self.c_c) * self.mu_eff
        ) * delta_mean

        rank_one = torch.outer(self.p_c, self.p_c)
        diffs = (selected - old_mean.unsqueeze(0)) / max(self.sigma, 1e-12)
        rank_mu = torch.zeros_like(self.C)
        for i in range(diffs.size(0)):
            rank_mu = rank_mu + weights[i] * torch.outer(diffs[i], diffs[i])

        self.C = (
            (1.0 - self.c_1 - self.c_mu) * self.C
            + self.c_1 * (rank_one + (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c) * self.C)
            + self.c_mu * rank_mu
        )
        self.C = 0.5 * (self.C + self.C.transpose(0, 1))

        self.sigma = self.sigma * math.exp(
            (self.c_sigma / self.d_sigma) * (norm_p_sigma / self.chi_n - 1.0)
        )
        self.sigma = float(min(max(self.sigma, 1e-8), self.bound))
