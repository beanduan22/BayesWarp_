from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import time
import torch

from bayeswarp.interpretability.saliency import compute_saliency
from bayeswarp.localization.region import localize_critical_region
from bayeswarp.bo.svgp_surrogate import SVGPSurrogate
from bayeswarp.mutation.grid_mutator import GridMutator
from bayeswarp.testing.objective import original_class, sorted_target_classes, bayeswarp_objective


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
    beta: float
    epsilon: float
    kappa: float
    n: int
    m: int
    per_class_budget: int
    max_target_classes: Optional[int] = None
    ablation: str = 'none'


class BayesWarpTester:
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

    def run_on_seed(self, x0: torch.Tensor) -> Dict[str, Any]:
        self.model.eval()
        x0 = x0.to(self.device)
        if x0.ndim == 3:
            x0 = x0.unsqueeze(0)
        og = original_class(self.model, x0)
        region_mask = self._region_mask(x0, og)
        mutator = GridMutator(
            image_shape=(x0.size(1), x0.size(2), x0.size(3)),
            region_mask=region_mask,
            n=self.cfg.n,
        )
        surrogate = SVGPSurrogate(dim=mutator.dim, m=self.cfg.m, device=self.device)
        failures = []
        start = time.perf_counter()

        targets = sorted_target_classes(self.model, x0, og)
        if self.cfg.max_target_classes is not None:
            targets = targets[: self.cfg.max_target_classes]

        for tg in targets:
            x = x0.clone()
            u = mutator.zero_params(self.device)
            for _ in range(self.cfg.per_class_budget):
                fx = bayeswarp_objective(self.model, x, og, tg)
                surrogate.add_observation(u, fx.view(1))
                candidates = mutator.sample_candidate_deltas(u, self.cfg.S)

                if self.cfg.ablation != 'no_bayesian':
                    surrogate.fit_step()
                    mu, sigma = surrogate.predict(candidates)
                    scores = mu + self.cfg.kappa * sigma
                    u_new = candidates[int(scores.argmax().item())]
                else:
                    u_new = candidates[torch.randint(0, candidates.size(0), (1,)).item()]

                x_new = mutator.apply(x.squeeze(0), u_new, eta=self.cfg.eta).unsqueeze(0)
                fx_new = bayeswarp_objective(self.model, x_new, og, tg)
                if abs(float((fx_new - fx).item())) <= self.cfg.epsilon:
                    x_new = x_new + self.cfg.beta * torch.randn_like(x_new)
                    pmin = float(x0.min().item())
                    pmax = float(x0.max().item())
                    lo = pmin - self.cfg.eta * (pmax - pmin)
                    hi = pmax + self.cfg.eta * (pmax - pmin)
                    x_new = x_new.clamp(lo, hi)

                pred_new = int(self.model(x_new).argmax(dim=1).item())
                if pred_new != og:
                    failures.append({
                        'x': x_new.detach().cpu(),
                        'target_class': int(tg),
                        'pred': pred_new,
                        'og': int(og),
                    })
                x = x_new.detach()
                u = u_new.detach()

        elapsed = time.perf_counter() - start
        return {
            'failures': failures,
            'og': int(og),
            'time_sec': float(elapsed),
            'region_mask': region_mask.detach().cpu(),
        }
