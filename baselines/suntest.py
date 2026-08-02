from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.common import Baseline, BudgetedOracle, BudgetExhausted, clip_to_seed_range


def tarantula(a_as: np.ndarray, a_af: np.ndarray, a_is: np.ndarray, a_if: np.ndarray) -> np.ndarray:
    failed = a_af / np.maximum(1e-12, a_af + a_if)
    passed = a_as / np.maximum(1e-12, a_as + a_is)
    return failed / np.maximum(1e-12, failed + passed)


def ochiai(a_as: np.ndarray, a_af: np.ndarray, a_is: np.ndarray, a_if: np.ndarray) -> np.ndarray:
    return a_af / np.sqrt(np.maximum(1e-12, (a_af + a_if) * (a_af + a_as)))


def dstar(a_as: np.ndarray, a_af: np.ndarray, a_is: np.ndarray, a_if: np.ndarray, star: int = 3) -> np.ndarray:
    return (a_af ** star) / np.maximum(1e-12, a_as + a_if)


SBFL_FORMULAS = {'tarantula': tarantula, 'ochiai': ochiai, 'dstar': dstar}


class SUNTest(Baseline):
    name = 'suntest'

    OPERATORS = (
        'pixel_change', 'salt_pepper', 'gaussian_noise', 'multiplicative_noise',
        'translation', 'brightness', 'rotation', 'median_blur', 'average_blur', 'gaussian_blur',
    )

    def __init__(self, model, device, cfg: Dict):
        super().__init__(model, device, cfg)
        self.num_intervals = int(cfg.get('b', 1000))
        self.top_intervals = int(cfg.get('K', 100))
        self.lam = float(cfg.get('lam', 0.1))
        self.q = float(cfg.get('q', 0.8))
        self.formula = str(cfg.get('formula', 'dstar'))
        self.star = int(cfg.get('star', 3))
        self.eps = float(cfg.get('eps', 1e-7))
        self.localization_samples = int(cfg.get('localization_samples', 2000))
        self.eta = float(cfg.get('eta', 0.1))

        self.final_linear = self._final_linear()
        self.suspicious: Optional[torch.Tensor] = None

        self.selected = np.zeros(len(self.OPERATORS))
        self.triggered = np.zeros(len(self.OPERATORS))
        self.improved = np.zeros(len(self.OPERATORS))
        self.fault_types: List[set] = [set() for _ in self.OPERATORS]

    def _final_linear(self) -> nn.Linear:
        linears = [m for m in self.model.modules() if isinstance(m, nn.Linear)]
        if not linears:
            raise ValueError('SUNTest needs a final Linear layer to read penultimate features.')
        return linears[-1]

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        captured: Dict[str, torch.Tensor] = {}

        def pre_hook(_m, inputs):
            captured['value'] = inputs[0]

        handle = self.final_linear.register_forward_pre_hook(pre_hook)
        logits = self.model(x)
        handle.remove()
        return logits, captured['value'].detach().flatten(1)

    def penultimate(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(x)[1]

    def prepare(self, train_dataset) -> None:
        n = min(self.localization_samples, len(train_dataset))
        generator = torch.Generator().manual_seed(0)
        indices = torch.randperm(len(train_dataset), generator=generator)[:n].tolist()

        outputs: List[torch.Tensor] = []
        correct: List[bool] = []
        with torch.no_grad():
            for idx in indices:
                x, y = train_dataset[idx]
                logits, feats = self.forward_with_features(x.unsqueeze(0).to(self.device))
                outputs.append(feats.squeeze(0).cpu())
                correct.append(int(logits.argmax(dim=1).item()) == int(y))

        acts = torch.stack(outputs)
        is_correct = torch.tensor(correct)

        thresholds = torch.zeros(acts.size(1))
        for j in range(acts.size(1)):
            column = acts[:, j]
            lo, hi = float(column.min()), float(column.max())
            if hi - lo < 1e-12:
                thresholds[j] = lo
                continue
            counts = torch.histc(column, bins=self.num_intervals, min=lo, max=hi)
            k = min(self.top_intervals, self.num_intervals)
            top = torch.topk(counts, k).indices
            width = (hi - lo) / self.num_intervals
            lowers = lo + top.float() * width
            uppers = lowers + width
            thresholds[j] = float((lowers + uppers).sum() / (2 * k))

        activated = acts > thresholds.unsqueeze(0)
        a_af = (activated & ~is_correct.unsqueeze(1)).sum(dim=0).numpy().astype(np.float64)
        a_as = (activated & is_correct.unsqueeze(1)).sum(dim=0).numpy().astype(np.float64)
        a_if = (~activated & ~is_correct.unsqueeze(1)).sum(dim=0).numpy().astype(np.float64)
        a_is = (~activated & is_correct.unsqueeze(1)).sum(dim=0).numpy().astype(np.float64)

        if self.formula == 'dstar':
            scores = dstar(a_as, a_af, a_is, a_if, self.star)
        else:
            scores = SBFL_FORMULAS[self.formula](a_as, a_af, a_is, a_if)

        num_suspicious = max(1, int(len(scores) * self.lam))
        order = np.argsort(scores)[::-1][:num_suspicious]
        self.suspicious = torch.tensor(order.copy(), dtype=torch.long)

    def _mutate(self, x: torch.Tensor, op: str, span: float, rng: torch.Generator) -> torch.Tensor:
        if op == 'pixel_change':
            out = x.clone()
            mask = torch.rand(x.shape, generator=rng) < 0.01
            out[mask.to(x.device)] = float(torch.rand(1, generator=rng).item()) * span + float(x.min())
            return out
        if op == 'salt_pepper':
            out = x.clone()
            r = torch.rand(x.shape, generator=rng).to(x.device)
            out[r < 0.01] = float(x.min())
            out[r > 0.99] = float(x.max())
            return out
        if op == 'gaussian_noise':
            return x + torch.randn(x.shape, generator=rng).to(x.device) * 0.05 * span
        if op == 'multiplicative_noise':
            return x * (1.0 + torch.randn(x.shape, generator=rng).to(x.device) * 0.05)
        if op == 'brightness':
            delta = (float(torch.rand(1, generator=rng).item()) * 0.2 - 0.1) * span
            return x + delta
        if op in ('translation', 'rotation'):
            n = x.size(0)
            theta = torch.zeros(n, 2, 3, device=x.device)
            theta[:, 0, 0] = 1.0
            theta[:, 1, 1] = 1.0
            if op == 'translation':
                theta[:, 0, 2] = (float(torch.rand(1, generator=rng).item()) * 2 - 1) * 0.1
                theta[:, 1, 2] = (float(torch.rand(1, generator=rng).item()) * 2 - 1) * 0.1
            else:
                deg = (float(torch.rand(1, generator=rng).item()) * 2 - 1) * 30.0
                rad = torch.tensor(deg * 3.141592653589793 / 180.0)
                cos, sin = torch.cos(rad), torch.sin(rad)
                theta[:, 0, 0], theta[:, 0, 1] = cos, -sin
                theta[:, 1, 0], theta[:, 1, 1] = sin, cos
            grid = F.affine_grid(theta, list(x.shape), align_corners=False)
            return F.grid_sample(x, grid, align_corners=False, padding_mode='border')
        if op in ('median_blur', 'average_blur', 'gaussian_blur'):
            k = 3
            pad = k // 2
            padded = F.pad(x, (pad,) * 4, mode='reflect')
            if op == 'median_blur':
                patches = padded.unfold(2, k, 1).unfold(3, k, 1).contiguous()
                return patches.view(*patches.shape[:4], -1).median(dim=-1).values
            if op == 'average_blur':
                weight = torch.ones(x.size(1), 1, k, k, device=x.device) / (k * k)
                return F.conv2d(padded, weight, groups=x.size(1))
            base = torch.tensor([1.0, 2.0, 1.0], device=x.device)
            kernel = torch.outer(base, base)
            kernel = (kernel / kernel.sum()).expand(x.size(1), 1, k, k).contiguous()
            return F.conv2d(padded, kernel, groups=x.size(1))
        return x

    def _rewards(self) -> np.ndarray:
        return (self.improved / (self.selected + self.eps)) * (
            np.array([len(s) for s in self.fault_types]) / (self.triggered + self.eps)
        )

    def _select_operator(self, current: Optional[int], rng: np.random.Generator) -> int:
        if current is None:
            return int(rng.integers(len(self.OPERATORS)))
        rewards = self._rewards()
        ranks = np.argsort(np.argsort(-rewards))
        candidate = int(rng.integers(len(self.OPERATORS)))
        p = 1.0 / len(self.OPERATORS)
        if ranks[candidate] <= ranks[current]:
            return candidate
        accept = min(1.0, (1.0 - p) ** (ranks[candidate] - ranks[current]))
        return candidate if rng.random() < accept else current

    def _fitness(self, feats: torch.Tensor, found: Optional[torch.Tensor], rng: np.random.Generator) -> float:
        idx = self.suspicious
        keep = max(1, int(round(len(idx) * self.q)))
        subset = idx[torch.from_numpy(rng.permutation(len(idx))[:keep].copy())]
        f = float(feats[subset].sum().item())
        if found is None or found.size(0) == 0:
            return f
        g = float((found - feats.unsqueeze(0)).norm(p=2, dim=1).min().item())
        return f + g

    def generate(self, x0: torch.Tensor, oracle: BudgetedOracle) -> None:
        if self.suspicious is None:
            raise RuntimeError('SUNTest requires prepare(train_dataset) before generation.')

        rng = np.random.default_rng(abs(hash((int(oracle.og), int(x0.numel())))) % (2 ** 32))
        torch_rng = torch.Generator().manual_seed(int(rng.integers(2 ** 31)))
        x0_img = x0.squeeze(0)
        span = float(x0.max().item() - x0.min().item()) or 1.0

        current = x0.clone()
        current_feats = self.penultimate(current).squeeze(0)
        found: Optional[torch.Tensor] = None
        f_max = self._fitness(current_feats, found, rng)
        op_idx: Optional[int] = None

        while not oracle.exhausted:
            op_idx = self._select_operator(op_idx, rng)
            op = self.OPERATORS[op_idx]
            self.selected[op_idx] += 1

            mutant = self._mutate(current, op, span, torch_rng)
            mutant = clip_to_seed_range(mutant, x0_img, self.eta)

            with torch.no_grad():
                logits, features = self.forward_with_features(mutant)
            try:
                conf = oracle.record(mutant, logits, target_class=-1)
            except BudgetExhausted:
                return
            label = int(conf.argmax().item())
            feats = features.squeeze(0)

            if label != oracle.og:

                self.triggered[op_idx] += 1
                self.fault_types[op_idx].add((int(oracle.og), label))
                row = feats.detach().unsqueeze(0)
                found = row if found is None else torch.cat([found, row], dim=0)
            else:
                fitness = self._fitness(feats, found, rng)
                if fitness > f_max:
                    f_max = fitness
                    current = mutant.detach()
                    self.improved[op_idx] += 1
