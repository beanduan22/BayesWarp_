from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import time

import torch

from bayeswarp.interpretability.saliency import compute_saliency
from bayeswarp.localization.region import localize_critical_region
from bayeswarp.bo.svgp_surrogate import SVGPSurrogate
from bayeswarp.mutation.grid_mutator import GridMutator
from bayeswarp.testing.objective import (
    allocate_budgets,
    margin_and_prediction,
    rank_target_classes,
    softmax_confidences,
)


@dataclass
class BayesWarpConfig:
    saliency_method: str
    alpha: float
    area_min: int
    tau_iou: float
    d_max: float
    rho: float
    S: int
    eta: float
    epsilon: float
    kappa: float
    r: float
    n: int
    m: int
    budget: int
    beta_min: float = 0.01
    beta_max: float = 0.05
    max_target_classes: Optional[int] = None
    ablation: str = 'none'


class BayesWarpTester:
    """Bayesian-guided white-box testing of a single input seed.

    For each seed the critical region and its grid parameterization are computed
    once and reused across all target-specific searches. The per-seed budget is
    divided across the confidence-ranked target classes; every forward
    evaluation of a newly constructed input counts against it.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device, cfg: BayesWarpConfig):
        self.model = model
        self.device = device
        self.cfg = cfg

    def _region_mask(self, x: torch.Tensor, og: int) -> torch.Tensor:
        if self.cfg.ablation == 'no_localization':
            return torch.ones(x.shape[-2:], device=self.device)
        sal = compute_saliency(self.model, x, og, self.cfg.saliency_method)
        region_mask, _ = localize_critical_region(
            sal.detach().cpu().numpy(),
            alpha=self.cfg.alpha,
            area_min=self.cfg.area_min,
            tau_iou=self.cfg.tau_iou,
            d_max=self.cfg.d_max,
            rho=self.cfg.rho,
        )
        return torch.from_numpy(region_mask).float().to(self.device)

    def _select_delta(self, surrogate: SVGPSurrogate, u: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        if self.cfg.ablation == 'no_bayesian':

            idx = int(torch.randint(0, deltas.size(0), (1,)).item())
            return deltas[idx]
        surrogate.fit_step()
        mu, sigma = surrogate.predict(u.unsqueeze(0) + deltas)
        scores = mu + self.cfg.kappa * sigma
        return deltas[int(scores.argmax().item())]

    def _evaluate(
        self,
        u: torch.Tensor,
        x0_img: torch.Tensor,
        og: int,
        tg: int,
        mutator: GridMutator,
        surrogate: SVGPSurrogate,
        failures: List[Dict[str, Any]],
    ) -> float:
        """Reconstruct x(u), spend one target-model evaluation, and record it.

        The input is added to the target-specific observation set and checked by
        the prediction-inconsistency oracle. A case is recorded whenever the
        top-1 prediction differs from og, regardless of whether it equals tg.
        """
        x = mutator.reconstruct(u, x0_img).unsqueeze(0)
        margin, pred = margin_and_prediction(self.model, x, og, tg)
        surrogate.add_observation(u, margin)
        if pred != og:
            failures.append({
                'x': x.detach().cpu(),
                'target_class': int(tg),
                'pred': int(pred),
                'og': int(og),
            })
        return margin

    def _search_target(
        self,
        x0_img: torch.Tensor,
        seed_margin: float,
        og: int,
        tg: int,
        budget: int,
        mutator: GridMutator,
        failures: List[Dict[str, Any]],
    ) -> None:

        surrogate = SVGPSurrogate(
            dim=mutator.dim,
            m=self.cfg.m,
            device=self.device,
            bound=self.cfg.r,
        )
        u = mutator.zero_params(self.device)

        fx = seed_margin
        surrogate.add_observation(u, fx)

        used = 0
        while used < budget:
            deltas = mutator.sample_deltas(self.cfg.S, self.device)
            delta_u = self._select_delta(surrogate, u, deltas)
            u_tilde = mutator.clip_u(u + delta_u)
            f_tilde = self._evaluate(u_tilde, x0_img, og, tg, mutator, surrogate, failures)
            used += 1

            if abs(f_tilde - fx) <= self.cfg.epsilon and used < budget:
                beta = float(torch.empty(1).uniform_(self.cfg.beta_min, self.cfg.beta_max).item())
                xi = torch.randn_like(u_tilde) * beta
                u_next = mutator.clip_u(u_tilde + xi)
                f_next = self._evaluate(u_next, x0_img, og, tg, mutator, surrogate, failures)
                used += 1
            else:
                u_next, f_next = u_tilde, f_tilde

            u, fx = u_next.detach(), f_next

    def run_on_seed(self, x0: torch.Tensor) -> Dict[str, Any]:
        self.model.eval()
        x0 = x0.to(self.device)
        if x0.ndim == 3:
            x0 = x0.unsqueeze(0)

        start = time.perf_counter()

        conf0 = softmax_confidences(self.model, x0).squeeze(0)
        og = int(conf0.argmax().item())
        region_mask = self._region_mask(x0, og)
        mutator = GridMutator(
            image_shape=(x0.size(1), x0.size(2), x0.size(3)),
            region_mask=region_mask,
            n=self.cfg.n,
            r=self.cfg.r,
            eta=self.cfg.eta,
        )

        targets = rank_target_classes(conf0, og)
        if self.cfg.max_target_classes is not None:
            targets = targets[: self.cfg.max_target_classes]
        budgets = allocate_budgets(self.cfg.budget, len(targets))

        failures: List[Dict[str, Any]] = []
        x0_img = x0.squeeze(0)
        for tg, budget in zip(targets, budgets):
            seed_margin = float(conf0[tg].item() - conf0[og].item())
            self._search_target(x0_img, seed_margin, og, tg, budget, mutator, failures)

        elapsed = time.perf_counter() - start
        return {
            'failures': failures,
            'og': int(og),
            'time_sec': float(elapsed),
            'region_mask': region_mask.detach().cpu(),
        }
