from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time

import torch

from bayeswarp.interpretability.saliency import compute_saliency
from bayeswarp.localization.region import localize_critical_region
from bayeswarp.bo.svgp_surrogate import SVGPSurrogate
from bayeswarp.bo.exact_gp import ExactGPSurrogate
from bayeswarp.bo.acquisition import build_acquisition
from bayeswarp.search.cmaes import CMAES
from bayeswarp.mutation.grid_mutator import build_mutator
from bayeswarp.testing.objective import (
    allocate_budgets,
    margin_and_prediction,
    rank_target_classes,
    softmax_confidences,
)


STAGES = (
    'saliency',
    'region_grid',
    'candidate_sampling',
    'surrogate_fit',
    'acquisition',
    'target_eval',
)

ABLATIONS = ('none', 'no_localization', 'no_bayesian', 'no_merging', 'no_grid', 'no_noise')


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
    surrogate: str = 'svgp'
    acquisition: str = 'ucb'
    search: str = 'bayesian'
    cma_sigma0: float = 0.05
    cma_population: int = 8

    def __post_init__(self):
        if self.ablation not in ABLATIONS:
            raise ValueError(f'Unsupported ablation: {self.ablation}')
        if self.search not in ('bayesian', 'cmaes'):
            raise ValueError(f'Unsupported search strategy: {self.search}')
        if self.surrogate not in ('svgp', 'exact_gp'):
            raise ValueError(f'Unsupported surrogate: {self.surrogate}')


@dataclass
class SeedState:
    used: int = 0
    qff: Optional[int] = None
    failures: List[Dict[str, Any]] = field(default_factory=list)
    timers: Dict[str, float] = field(default_factory=lambda: {s: 0.0 for s in STAGES})


class BayesWarpTester:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        cfg: BayesWarpConfig,
        p_min: torch.Tensor,
        p_max: torch.Tensor,
    ):
        self.model = model
        self.device = device
        self.cfg = cfg
        self.p_min = p_min
        self.p_max = p_max
        self.acquisition = build_acquisition(cfg.acquisition)

    def _region_mask(self, x: torch.Tensor, og: int, state: SeedState) -> torch.Tensor:
        if self.cfg.ablation == 'no_localization':
            return torch.ones(x.shape[-2:], device=self.device)
        start = time.perf_counter()
        sal = compute_saliency(self.model, x, og, self.cfg.saliency_method)
        state.timers['saliency'] += time.perf_counter() - start
        region_mask, _ = localize_critical_region(
            sal.detach().cpu().numpy(),
            alpha=self.cfg.alpha,
            area_min=self.cfg.area_min,
            tau_iou=self.cfg.tau_iou,
            d_max=self.cfg.d_max,
            rho=self.cfg.rho,
            merge=self.cfg.ablation != 'no_merging',
        )
        return torch.from_numpy(region_mask).float().to(self.device)

    def _build_surrogate(self, dim: int, bound: float):
        if self.cfg.surrogate == 'exact_gp':
            return ExactGPSurrogate(dim=dim, device=self.device, bound=bound)
        return SVGPSurrogate(dim=dim, m=self.cfg.m, device=self.device, bound=bound)

    def _select_delta(self, surrogate, u: torch.Tensor, deltas: torch.Tensor, best: float, state: SeedState) -> torch.Tensor:
        if self.cfg.ablation == 'no_bayesian':
            idx = int(torch.randint(0, deltas.size(0), (1,)).item())
            return deltas[idx]

        start = time.perf_counter()
        surrogate.fit_step()
        state.timers['surrogate_fit'] += time.perf_counter() - start

        start = time.perf_counter()
        mu, sigma = surrogate.predict(u.unsqueeze(0) + deltas)
        scores = self.acquisition(mu, sigma, self.cfg.kappa, best)
        choice = deltas[int(scores.argmax().item())]
        state.timers['acquisition'] += time.perf_counter() - start
        return choice

    def _evaluate(
        self,
        u: torch.Tensor,
        x0_img: torch.Tensor,
        og: int,
        tg: int,
        mutator,
        surrogate,
        state: SeedState,
    ) -> float:
        start = time.perf_counter()
        x = mutator.reconstruct(u, x0_img).unsqueeze(0)
        margin, pred = margin_and_prediction(self.model, x, og, tg)
        state.timers['target_eval'] += time.perf_counter() - start

        state.used += 1
        if surrogate is not None:
            surrogate.add_observation(u, margin)
        if pred != og:
            if state.qff is None:
                state.qff = state.used
            state.failures.append({
                'x': x.detach().cpu(),
                'target_class': int(tg),
                'pred': int(pred),
                'og': int(og),
                'evaluation_index': int(state.used),
            })
        return margin

    def _search_target_bayesian(
        self,
        x0_img: torch.Tensor,
        seed_margin: float,
        og: int,
        tg: int,
        budget: int,
        mutator,
        state: SeedState,
    ) -> None:
        surrogate = self._build_surrogate(mutator.dim, mutator.u_bound)
        u = mutator.zero_params(self.device)

        fx = seed_margin
        best = seed_margin
        surrogate.add_observation(u, fx)

        used = 0
        while used < budget:
            start = time.perf_counter()
            deltas = mutator.sample_deltas(self.cfg.S, self.device)
            state.timers['candidate_sampling'] += time.perf_counter() - start

            delta_u = self._select_delta(surrogate, u, deltas, best, state)
            u_tilde = mutator.clip_u(u + delta_u)
            f_tilde = self._evaluate(u_tilde, x0_img, og, tg, mutator, surrogate, state)
            used += 1
            best = max(best, f_tilde)

            stagnated = abs(f_tilde - fx) <= self.cfg.epsilon
            if stagnated and self.cfg.ablation != 'no_noise' and used < budget:
                beta = float(torch.empty(1).uniform_(self.cfg.beta_min, self.cfg.beta_max).item())
                xi = torch.randn_like(u_tilde) * beta
                u_next = mutator.clip_u(u_tilde + xi)
                f_next = self._evaluate(u_next, x0_img, og, tg, mutator, surrogate, state)
                used += 1
                best = max(best, f_next)
            else:
                u_next, f_next = u_tilde, f_tilde

            u, fx = u_next.detach(), f_next

    def _search_target_cmaes(
        self,
        x0_img: torch.Tensor,
        seed_margin: float,
        og: int,
        tg: int,
        budget: int,
        mutator,
        state: SeedState,
    ) -> None:
        optimizer = CMAES(
            dim=mutator.dim,
            device=self.device,
            bound=mutator.u_bound,
            sigma0=self.cfg.cma_sigma0,
            population=self.cfg.cma_population,
        )
        used = 0
        while used < budget:
            start = time.perf_counter()
            offspring = optimizer.ask()
            state.timers['candidate_sampling'] += time.perf_counter() - start

            solutions: List[torch.Tensor] = []
            values: List[float] = []
            for candidate in offspring:
                if used >= budget:
                    break
                u = mutator.clip_u(candidate.to(self.device))
                value = self._evaluate(u, x0_img, og, tg, mutator, None, state)
                used += 1
                solutions.append(u)
                values.append(value)
            optimizer.tell(solutions, values)

    def run_on_seed(self, x0: torch.Tensor) -> Dict[str, Any]:
        self.model.eval()
        x0 = x0.to(self.device)
        if x0.ndim == 3:
            x0 = x0.unsqueeze(0)

        state = SeedState()
        start = time.perf_counter()

        conf0 = softmax_confidences(self.model, x0).squeeze(0)
        og = int(conf0.argmax().item())

        region_mask = self._region_mask(x0, og, state)

        grid_start = time.perf_counter()
        mutator = build_mutator(
            image_shape=(x0.size(1), x0.size(2), x0.size(3)),
            region_mask=region_mask,
            p_min=self.p_min,
            p_max=self.p_max,
            n=self.cfg.n,
            r=self.cfg.r,
            eta=self.cfg.eta,
            parameterization='pixel' if self.cfg.ablation == 'no_grid' else 'grid',
        )
        state.timers['region_grid'] += time.perf_counter() - grid_start

        targets = rank_target_classes(conf0, og)
        if self.cfg.max_target_classes is not None:
            targets = targets[: self.cfg.max_target_classes]
        budgets = allocate_budgets(self.cfg.budget, len(targets))

        x0_img = x0.squeeze(0)
        search = self._search_target_cmaes if self.cfg.search == 'cmaes' else self._search_target_bayesian
        for tg, budget in zip(targets, budgets):
            if budget <= 0:
                continue
            seed_margin = float(conf0[tg].item() - conf0[og].item())
            search(x0_img, seed_margin, og, tg, budget, mutator, state)

        elapsed = time.perf_counter() - start
        return {
            'failures': state.failures,
            'og': int(og),
            'time_sec': float(elapsed),
            'evaluations_used': int(state.used),
            'qff': int(state.qff) if state.qff is not None else int(self.cfg.budget),
            'stage_time_sec': {k: float(v) for k, v in state.timers.items()},
            'region_mask': region_mask.detach().cpu(),
            'mutation_dim': int(mutator.dim),
        }
