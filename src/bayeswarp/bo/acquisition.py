from __future__ import annotations
import math

import torch


def upper_confidence_bound(mu: torch.Tensor, sigma: torch.Tensor, kappa: float, best: float) -> torch.Tensor:
    return mu + kappa * sigma


def expected_improvement(mu: torch.Tensor, sigma: torch.Tensor, kappa: float, best: float, xi: float = 0.0) -> torch.Tensor:
    sigma = sigma.clamp_min(1e-12)
    improvement = mu - best - xi
    z = improvement / sigma
    normal = torch.distributions.Normal(
        torch.zeros((), device=mu.device, dtype=mu.dtype),
        torch.ones((), device=mu.device, dtype=mu.dtype),
    )
    cdf = normal.cdf(z)
    pdf = torch.exp(normal.log_prob(z))
    return improvement * cdf + sigma * pdf


ACQUISITIONS = {
    'ucb': upper_confidence_bound,
    'ei': expected_improvement,
}


def build_acquisition(name: str):
    key = name.lower()
    if key not in ACQUISITIONS:
        raise ValueError(f'Unsupported acquisition function: {name}')
    return ACQUISITIONS[key]
