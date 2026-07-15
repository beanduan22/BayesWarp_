"""NSGen. Reimplementation of Huang et al., "Neuron Semantic-Guided Test
Generation for Deep Neural Networks Fuzzing", ACM TOSEM.

The reference derives neuron semantics from MILAN captions compared with the
CLIP text encoder; those artifacts are not distributed, so this compares the
underlying semantic descriptor (decision-path units and pseudo-labels) with
Jaccard similarity. Style-transfer mutations are omitted.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.common import Baseline, BudgetedOracle, BudgetExhausted, clip_to_seed_range


DEFAULT_ALPHA_BY_DATASET = {'imagenet': 0.72, 'cifar10': 0.81, 'mnist': 0.81}

DEFAULT_PSEUDO_TOPK = {'imagenet': 3, 'cifar10': 1, 'mnist': 1}


def _affine(x: torch.Tensor, kind: str, rng: torch.Generator) -> torch.Tensor:
    """G operators: translation, scale, rotation."""
    n, c, h, w = x.shape
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    if kind == 'translation':
        offsets = torch.tensor([-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0])
        dx = offsets[torch.randint(len(offsets), (1,), generator=rng)].item() / max(1, w) * 2
        dy = offsets[torch.randint(len(offsets), (1,), generator=rng)].item() / max(1, h) * 2
        theta[:, 0, 2], theta[:, 1, 2] = dx, dy
    elif kind == 'scale':
        factors = torch.tensor([0.80, 0.85, 0.90, 0.95])
        s = factors[torch.randint(len(factors), (1,), generator=rng)].item()
        theta[:, 0, 0], theta[:, 1, 1] = 1.0 / s, 1.0 / s
    elif kind == 'rotation':
        deg = float(torch.randint(-30, 30, (1,), generator=rng).item())
        rad = torch.tensor(deg * 3.141592653589793 / 180.0)
        cos, sin = torch.cos(rad), torch.sin(rad)
        theta[:, 0, 0], theta[:, 0, 1] = cos, -sin
        theta[:, 1, 0], theta[:, 1, 1] = sin, cos
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode='border')


def _pixel(x: torch.Tensor, kind: str, span: float, rng: torch.Generator) -> torch.Tensor:
    """P operators: contrast, brightness, blur."""
    if kind == 'contrast':
        factor = 0.8 + 1.2 * float(torch.rand(1, generator=rng).item())
        return (x - x.mean()) * factor + x.mean()
    if kind == 'brightness':
        delta = (10.0 + 60.0 * float(torch.rand(1, generator=rng).item())) / 255.0 * span
        sign = 1.0 if torch.rand(1, generator=rng).item() > 0.5 else -1.0
        return x + sign * delta
    if kind == 'blur':
        k = int(torch.randint(0, 3, (1,), generator=rng).item()) * 2 + 3
        pad = k // 2
        weight = torch.ones(x.size(1), 1, k, k, device=x.device) / (k * k)
        return F.conv2d(F.pad(x, (pad,) * 4, mode='reflect'), weight, groups=x.size(1))
    return x


class NSGen(Baseline):
    name = 'nsgen'

    AFFINE_OPS = ('translation', 'scale', 'rotation')
    PIXEL_OPS = ('contrast', 'brightness', 'blur')

    def __init__(self, model, device, cfg: Dict):
        super().__init__(model, device, cfg)
        self.alpha_accept = float(cfg.get('alpha_accept', DEFAULT_ALPHA_BY_DATASET['cifar10']))
        self.alpha_frac = float(cfg.get('alpha_frac', 0.2))
        self.beta_frac = float(cfg.get('beta_frac', 0.4))
        self.try_num = int(cfg.get('try_num', 50))
        self.topk = int(cfg.get('topk', 1))
        self.pseudo_topk = int(cfg.get('pseudo_topk', 1))
        self.eta = float(cfg.get('eta', 0.1))
        self.layers = self._instrumented_layers()

    def _instrumented_layers(self) -> List[Tuple[str, nn.Module]]:
        layers = [
            (name, module)
            for name, module in self.model.named_modules()
            if len(list(module.children())) == 0 and isinstance(module, nn.Conv2d)
        ]
        if not layers:
            layers = [
                (name, module)
                for name, module in self.model.named_modules()
                if len(list(module.children())) == 0 and isinstance(module, nn.Linear)
            ]

        if len(layers) > 5:
            step = len(layers) / 5.0
            layers = [layers[min(len(layers) - 1, int(i * step))] for i in range(5)]
        return layers

    def semantics(self, x: torch.Tensor) -> Tuple[FrozenSet, torch.Tensor]:
        """Semantic descriptor of `x`: decision-path units plus pseudo-labels.

        The decision path is the top critical channel per instrumented layer,
        ranked by gradient x activation attribution to the predicted class. This
        stands in for the reference's LayerConductance attribution.

        Returns the descriptor set and the logits from the same forward pass.
        """
        acts: Dict[str, torch.Tensor] = {}
        handles = []
        for name, module in self.layers:
            def hook(_m, _i, out, n=name):
                acts[n] = out
                out.retain_grad()
            handles.append(module.register_forward_hook(hook))

        logits = self.model(x.clone().requires_grad_(True))
        label = int(logits.argmax(dim=1).item())
        self.model.zero_grad(set_to_none=True)
        logits[0, label].backward()
        for h in handles:
            h.remove()

        descriptor: List[Tuple] = []

        k_cls = min(self.pseudo_topk, logits.size(1))
        for c in torch.topk(logits.detach(), k_cls, dim=1).indices.squeeze(0).tolist():
            descriptor.append(('class', int(c)))

        for name, _ in self.layers:
            a = acts[name]
            if a.grad is None:
                continue
            attribution = (a.grad.detach() * a.detach()).abs()
            attribution = attribution.reshape(attribution.size(0), attribution.size(1), -1).mean(dim=2).squeeze(0)
            k = min(self.topk, attribution.numel())
            for idx in torch.topk(attribution, k).indices.tolist():
                descriptor.append((name, int(idx)))
        return frozenset(descriptor), logits.detach()

    @staticmethod
    def jaccard(a: FrozenSet, b: FrozenSet) -> float:
        union = len(a | b)
        return len(a & b) / union if union else 1.0

    def _mutate(self, x: torch.Tensor, state: int, span: float, rng: torch.Generator) -> Tuple[torch.Tensor, int]:
        """Two-state scheme: affine allowed only while state == 0."""
        if state == 0 and torch.rand(1, generator=rng).item() < 0.5:
            kind = self.AFFINE_OPS[int(torch.randint(len(self.AFFINE_OPS), (1,), generator=rng).item())]
            return _affine(x, kind, rng), 1
        kind = self.PIXEL_OPS[int(torch.randint(len(self.PIXEL_OPS), (1,), generator=rng).item())]
        return _pixel(x, kind, span, rng), state

    def _satisfies_constraint(self, x: torch.Tensor, x0: torch.Tensor, span: float) -> bool:
        """Metamorphic distance constraint from DeepHunter."""
        changed = int((x - x0).abs().gt(1e-6).sum().item())
        active = max(1, int(x0.gt(0).sum().item()))
        linf = float((x - x0).abs().max().item())
        if changed < self.alpha_frac * active:
            return linf <= span
        return linf <= self.beta_frac * span

    def generate(self, x0: torch.Tensor, oracle: BudgetedOracle) -> None:
        rng = torch.Generator().manual_seed(int(oracle.og) + 12345)
        x0_img = x0.squeeze(0)
        span = float(x0.max().item() - x0.min().item()) or 1.0
        seed_semantics, _ = self.semantics(x0)

        queue: List[Tuple[torch.Tensor, int]] = [(x0.clone(), 0)]
        while not oracle.exhausted:
            if not queue:
                queue.append((x0.clone(), 0))
            current, state = queue.pop(0)

            for _ in range(self.try_num):
                if oracle.exhausted:
                    return
                mutant, new_state = self._mutate(current, state, span, rng)
                mutant = clip_to_seed_range(mutant, x0_img, self.eta)
                if not self._satisfies_constraint(mutant, x0, span):
                    continue

                mutant_semantics, logits = self.semantics(mutant)
                try:
                    oracle.record(mutant, logits, target_class=-1)
                except BudgetExhausted:
                    return

                similarity = self.jaccard(mutant_semantics, seed_semantics)

                if 0.0 < similarity < self.alpha_accept:
                    queue.append((mutant.detach(), new_state))
                    break
