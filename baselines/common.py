from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


class BudgetExhausted(Exception):
    pass


@dataclass
class BudgetedOracle:
    model: torch.nn.Module
    budget: int
    og: int = -1
    used: int = 0
    qff: Optional[int] = None
    failures: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.used)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.budget

    @torch.no_grad()
    def seed_prediction(self, x0: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(x0), dim=1).squeeze(0)

    @torch.no_grad()
    def evaluate(self, x: torch.Tensor, target_class: int = -1) -> torch.Tensor:
        return self.record(x, self.model(x), target_class)

    def record(self, x: torch.Tensor, logits: torch.Tensor, target_class: int = -1) -> torch.Tensor:
        if self.exhausted:
            raise BudgetExhausted(f'budget of {self.budget} evaluations is exhausted')
        self.used += 1
        conf = torch.softmax(logits.detach(), dim=1).squeeze(0)
        pred = int(conf.argmax().item())
        if pred != self.og:
            if self.qff is None:
                self.qff = self.used
            self.failures.append({
                'x': x.detach().cpu(),
                'target_class': int(target_class),
                'pred': pred,
                'og': int(self.og),
                'evaluation_index': int(self.used),
            })
        return conf


class Baseline(ABC):
    name: str = 'baseline'

    def __init__(self, model: torch.nn.Module, device: torch.device, cfg: Dict[str, Any]):
        self.model = model
        self.device = device
        self.cfg = cfg

    def prepare(self, train_dataset) -> None:
        return None

    @abstractmethod
    def generate(self, x0: torch.Tensor, oracle: BudgetedOracle) -> None:
        raise NotImplementedError

    def run_on_seed(self, x0: torch.Tensor, budget: int) -> Dict[str, Any]:
        import time

        self.model.eval()
        x0 = x0.to(self.device)
        if x0.ndim == 3:
            x0 = x0.unsqueeze(0)

        start = time.perf_counter()
        oracle = BudgetedOracle(model=self.model, budget=budget)
        oracle.og = int(oracle.seed_prediction(x0).argmax().item())
        try:
            self.generate(x0, oracle)
        except BudgetExhausted:
            pass
        elapsed = time.perf_counter() - start

        return {
            'failures': oracle.failures,
            'og': int(oracle.og),
            'time_sec': float(elapsed),
            'evaluations_used': int(oracle.used),
            'qff': int(oracle.qff) if oracle.qff is not None else int(budget),
        }


def clip_to_seed_range(x: torch.Tensor, x_seed: torch.Tensor, eta: float = 0.1) -> torch.Tensor:
    pmin = float(x_seed.min().item())
    pmax = float(x_seed.max().item())
    span = pmax - pmin
    return x.clamp(pmin - eta * span, pmax + eta * span)


def named_leaf_layers(model: torch.nn.Module) -> List[tuple]:
    layers = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0 and isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            layers.append((name, module))
    return layers


def collect_activations(model: torch.nn.Module, x: torch.Tensor, layers: Optional[List[tuple]] = None) -> Dict[str, torch.Tensor]:
    layers = layers if layers is not None else named_leaf_layers(model)
    acts: Dict[str, torch.Tensor] = {}
    handles = []
    for name, module in layers:
        def hook(_m, _i, out, n=name):
            acts[n] = out
            if out.requires_grad:
                out.retain_grad()
        handles.append(module.register_forward_hook(hook))
    logits = model(x)
    for h in handles:
        h.remove()
    acts['__logits__'] = logits
    return acts
