from __future__ import annotations
import torch
import torch.nn.functional as F


@torch.no_grad()
def softmax_confidences(model, x: torch.Tensor) -> torch.Tensor:
    return F.softmax(model(x), dim=1)


@torch.no_grad()
def original_class(model, x: torch.Tensor) -> int:
    return int(softmax_confidences(model, x).argmax(dim=1).item())


@torch.no_grad()
def adaptive_lambda(conf_tg: float, conf_og: float) -> float:
    return conf_tg / (conf_tg + conf_og + 1e-12)


@torch.no_grad()
def bayeswarp_objective(model, x: torch.Tensor, og: int, tg: int) -> torch.Tensor:
    conf = softmax_confidences(model, x).squeeze(0)
    conf_tg = conf[tg]
    conf_og = conf[og]
    lam = adaptive_lambda(float(conf_tg.item()), float(conf_og.item()))
    return lam * conf_tg - (1.0 - lam) * conf_og


@torch.no_grad()
def sorted_target_classes(model, x: torch.Tensor, og: int):
    conf = softmax_confidences(model, x).squeeze(0)
    pairs = [(i, float(conf[i].item())) for i in range(conf.numel()) if i != og]
    pairs.sort(key=lambda z: z[1], reverse=True)
    return [i for i, _ in pairs]
