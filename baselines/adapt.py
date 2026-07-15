"""ADAPT. PyTorch port of Lee et al., "Effective White-Box Testing of Deep
Neural Networks with Adaptive Neuron-Selection Strategy", ISSTA 2020
(https://github.com/kupl/adapt, MIT).

Feature f7 is set from the 30-40% weight band; the reference indexes it with an
always-empty slice, leaving the feature dead.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from baselines.common import Baseline, BudgetedOracle, BudgetExhausted, clip_to_seed_range

NUM_FEATURES = 29


def _layer_type_index(module: nn.Module) -> int:
    """One-hot slot (features 10-16) for the layer's type."""
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        return 10
    if isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
        return 11
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ZeroPad2d)):
        return 12
    if isinstance(module, nn.Linear):
        return 13
    if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)):
        return 14
    return 16


class FeatureMatrix:
    """29-dimensional per-neuron features: 17 constant, 12 activation-dependent."""

    def __init__(self, layers: Sequence[Tuple[str, nn.Module]], neurons: Sequence[Tuple[int, int]]):
        self.layers = list(layers)
        self.neurons = list(neurons)
        n = len(self.neurons)
        self.const = np.zeros((n, NUM_FEATURES), dtype=np.float32)
        self.variable = np.zeros((n, NUM_FEATURES), dtype=np.float32)

        num_layers = max(1, len(self.layers) - 1)
        weights = np.zeros(n, dtype=np.float32)
        for i, (li, ni) in enumerate(self.neurons):
            name, module = self.layers[li]

            quartile = int((li / num_layers) * 4)
            self.const[i, min(quartile, 3)] = 1.0

            self.const[i, _layer_type_index(module)] = 1.0

            weight = getattr(module, 'weight', None)
            if weight is not None and ni < weight.shape[0]:
                weights[i] = float(weight.detach()[ni].mean().item())

        order = np.argsort(weights)[::-1]
        bands = [(4, 0.0, 0.1), (5, 0.1, 0.2), (6, 0.2, 0.3), (7, 0.3, 0.4), (8, 0.4, 0.5), (9, 0.5, 1.0)]
        for feature, lo, hi in bands:
            self.const[order[int(n * lo):int(n * hi)], feature] = 1.0

        self.covered_count = np.zeros(n, dtype=np.float32)
        self.objective_covered = np.zeros(n, dtype=np.float32)
        self.refresh()

    def refresh(self) -> None:
        self.variable[:] = 0.0

        self.variable[:, 17] = self.objective_covered

        self.variable[:, 18] = (self.covered_count < 1).astype(np.float32)

        n = len(self.neurons)
        order = np.argsort(self.covered_count)[::-1]
        for d in range(10):
            lo, hi = int(n * d * 0.1), int(n * (d + 1) * 0.1)
            self.variable[order[lo:hi], 19 + d] = 1.0

    def update(self, covered: np.ndarray, label_changed: bool) -> None:
        self.covered_count += covered.astype(np.float32)
        if label_changed:
            self.objective_covered = np.maximum(self.objective_covered, covered.astype(np.float32))
        self.refresh()

    def matrix(self) -> np.ndarray:
        return self.const + self.variable


def greedy_max_set(covereds: List[np.ndarray], n: int) -> List[int]:
    """Greedy max set cover: repeatedly take the record adding most new coverage."""
    selected: List[int] = []
    if not covereds:
        return selected
    pool = set(range(len(covereds)))
    accumulated = np.zeros_like(covereds[0], dtype=bool)
    while pool and len(selected) < n:
        best, best_gain = -1, 0
        for i in pool:
            gain = int(np.logical_and(covereds[i].astype(bool), ~accumulated).sum())
            if gain > best_gain:
                best, best_gain = i, gain
        if best < 0:
            break
        selected.append(best)
        accumulated |= covereds[best].astype(bool)
        pool.discard(best)
    return selected


class AdaptiveStrategy:
    """Genetic search over 29-dimensional neuron-selection strategy vectors."""

    def __init__(self, features: FeatureMatrix, rng: np.random.Generator,
                 bound: float = 5.0, size: int = 100, history: int = 300,
                 remainder: float = 0.5, sigma: float = 1.0):
        self.features = features
        self.rng = rng
        self.bound = bound
        self.size = size
        self.history = history
        self.remainder = remainder
        self.sigma = sigma
        self.records: List[Tuple[np.ndarray, np.ndarray]] = []
        self.queue: List[np.ndarray] = []
        self.vector = self._random_vector()
        self.strategy_covered = np.zeros(len(features.neurons), dtype=np.float32)

    def _random_vector(self) -> np.ndarray:
        return self.rng.uniform(-self.bound, self.bound, NUM_FEATURES).astype(np.float32)

    def select(self, k: int) -> List[Tuple[int, int]]:
        scores = self.features.matrix().dot(self.vector)
        k = min(k, len(scores))
        idx = np.argpartition(scores, -k)[-k:]
        return [self.features.neurons[i] for i in idx]

    def update(self, covered: np.ndarray, label_changed: bool) -> None:
        self.features.update(covered, label_changed)
        self.strategy_covered = np.maximum(self.strategy_covered, covered.astype(np.float32))

    def next(self) -> None:
        """Finish this episode and advance to the next strategy vector."""
        self.records.append((self.vector.copy(), self.strategy_covered.copy()))
        self.strategy_covered = np.zeros(len(self.features.neurons), dtype=np.float32)
        if self.queue:
            self.vector = self.queue.pop(0)
            return
        self.queue = self._regenerate()
        self.vector = self.queue.pop(0) if self.queue else self._random_vector()

    def _regenerate(self) -> List[np.ndarray]:
        recent = self.records[-self.history:]
        if len(recent) < 2:
            return [self._random_vector() for _ in range(self.size)]

        vectors = [v for v, _ in recent]
        covereds = [c for _, c in recent]
        n_parents = max(2, int(self.size * self.remainder))
        chosen = greedy_max_set(covereds, n_parents)
        if len(chosen) < n_parents:

            ranked = sorted(range(len(recent)), key=lambda i: float(covereds[i].mean()), reverse=True)
            for i in ranked:
                if len(chosen) >= n_parents:
                    break
                if i not in chosen:
                    chosen.append(i)
        parents = [vectors[i] for i in chosen]

        offspring: List[np.ndarray] = []
        repeats = int(np.ceil(1.0 / self.remainder))
        for _ in range(repeats):
            order = self.rng.permutation(len(parents))
            half = len(parents) // 2
            for a, b in zip(order[:half], order[half:2 * half]):
                left, right = parents[a], parents[b]
                coin = self.rng.random(NUM_FEATURES) < 0.5
                child = np.where(coin, left, right).astype(np.float32)
                child = child + self.rng.normal(0.0, self.sigma, NUM_FEATURES).astype(np.float32)
                offspring.append(np.clip(child, -self.bound, self.bound))
        if not offspring:
            offspring = [self._random_vector() for _ in range(self.size)]
        return offspring[: self.size]


class Adapt(Baseline):
    name = 'adapt'

    def __init__(self, model, device, cfg: Dict):
        super().__init__(model, device, cfg)
        self.k = int(cfg.get('k', 10))
        self.lr = float(cfg.get('lr', 0.1))
        self.trail = int(cfg.get('trail', 3))
        self.delta = float(cfg.get('delta', 0.5))
        self.class_weight = float(cfg.get('class_weight', 0.5))
        self.neuron_weight = float(cfg.get('neuron_weight', 0.5))
        self.theta = float(cfg.get('theta', 0.5))
        self.eta = float(cfg.get('eta', 0.1))
        self.layers = self._instrumented_layers()

    def _instrumented_layers(self) -> List[Tuple[str, nn.Module]]:
        layers = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0 and isinstance(module, (nn.Conv2d, nn.Linear)):
                layers.append((name, module))

        return layers[:-1] if len(layers) > 1 else layers

    def _forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Return per-layer channel-averaged activations and the logits."""
        acts: Dict[str, torch.Tensor] = {}
        handles = []
        for name, module in self.layers:
            def hook(_m, _i, out, n=name):
                acts[n] = out
            handles.append(module.register_forward_hook(hook))
        logits = self.model(x)
        for h in handles:
            h.remove()
        internals = []
        for name, _ in self.layers:
            a = acts[name]

            internals.append(a.reshape(a.size(0), a.size(1), -1).mean(dim=2).squeeze(0) if a.ndim > 2 else a.squeeze(0))
        return internals, logits

    @staticmethod
    def _coverage_vector(internals: Sequence[torch.Tensor], theta: float) -> np.ndarray:
        """Per-layer min-max normalization, then threshold."""
        covered = []
        for a in internals:
            lo, hi = a.min(), a.max()
            norm = (a - lo) / (hi - lo + 1e-6)
            covered.append((norm > theta).detach().cpu().numpy())
        return np.concatenate(covered) if covered else np.zeros(0, dtype=bool)

    def generate(self, x0: torch.Tensor, oracle: BudgetedOracle) -> None:
        rng = np.random.default_rng(abs(hash((int(oracle.og), int(x0.numel())))) % (2 ** 32))
        with torch.no_grad():
            internals0, _ = self._forward(x0)
        neurons = [(li, ni) for li, a in enumerate(internals0) for ni in range(a.numel())]
        features = FeatureMatrix(self.layers, neurons)
        strategy = AdaptiveStrategy(features, rng)

        x0_img = x0.squeeze(0)
        orig_norm = float(x0.norm().item()) + 1e-12

        while not oracle.exhausted:
            worklist: List[torch.Tensor] = [x0.clone()]
            while worklist and not oracle.exhausted:
                current = worklist.pop(0)
                selected = strategy.select(self.k)
                for _ in range(self.trail):
                    if oracle.exhausted:
                        return
                    x = current.clone().detach().requires_grad_(True)
                    internals, logits = self._forward(x)
                    activation = torch.stack([internals[li][ni] for li, ni in selected]).sum()
                    loss = self.neuron_weight * activation - self.class_weight * logits[0, oracle.og]
                    self.model.zero_grad(set_to_none=True)
                    grad = torch.autograd.grad(loss, x)[0]

                    current = (current + self.lr * grad).detach()
                    current = clip_to_seed_range(current, x0_img, self.eta)

                    with torch.no_grad():
                        internals_new, logits_new = self._forward(current)
                    try:
                        conf = oracle.record(current, logits_new, target_class=-1)
                    except BudgetExhausted:
                        return
                    label = int(conf.argmax().item())
                    covered = self._coverage_vector(internals_new, self.theta)
                    before = float(strategy.features.covered_count.astype(bool).mean())
                    strategy.update(covered, label_changed=label != oracle.og)
                    after = float(strategy.features.covered_count.astype(bool).mean())

                    distance = float((current - x0).norm().item()) / orig_norm
                    if after > before and distance < self.delta:
                        worklist.append(current.clone())
            strategy.next()
