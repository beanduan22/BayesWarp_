from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn.functional as F


@torch.no_grad()
def softmax_confidences(model, x: torch.Tensor) -> torch.Tensor:
    return F.softmax(model(x), dim=1)


@torch.no_grad()
def margin_and_prediction(model, x: torch.Tensor, og: int, tg: int) -> Tuple[float, int]:
    conf = softmax_confidences(model, x).squeeze(0)
    margin = float(conf[tg].item() - conf[og].item())
    pred = int(conf.argmax().item())
    return margin, pred


def rank_target_classes(conf: torch.Tensor, og: int) -> List[int]:
    pairs = [(i, float(conf[i].item())) for i in range(conf.numel()) if i != og]
    pairs.sort(key=lambda z: z[1], reverse=True)
    return [i for i, _ in pairs]


def allocate_budgets(total_budget: int, num_targets: int) -> List[int]:
    if num_targets <= 0:
        return []
    base = total_budget // num_targets
    remainder = total_budget % num_targets
    return [base + (1 if k < remainder else 0) for k in range(num_targets)]
