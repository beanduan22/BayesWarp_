from __future__ import annotations
from typing import Dict, List, Tuple
import torch
import torch.nn as nn


SUPPORTED_LEAF_TYPES = (nn.Conv2d, nn.Linear)


def _collect_named_leaf_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    modules = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0 and isinstance(module, SUPPORTED_LEAF_TYPES):
            modules.append((name, module))
    return modules


def _forward_collect(model: nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    acts = {}
    handles = []
    for name, module in _collect_named_leaf_modules(model):
        handles.append(module.register_forward_hook(lambda m, i, o, n=name: acts.__setitem__(n, o.detach())))
    _ = model(x)
    for h in handles:
        h.remove()
    return acts


def neuron_coverage(model: nn.Module, images: List[torch.Tensor], threshold: float = 0.0) -> float:
    if len(images) == 0:
        return 0.0
    total = 0
    acts0 = _forward_collect(model, images[0])
    for a in acts0.values():
        total += a[0].numel()
    mask_sum = None
    for x in images:
        acts = _forward_collect(model, x)
        flat = torch.cat([a.flatten() for a in acts.values()])
        curr = (flat > threshold).float()
        mask_sum = curr if mask_sum is None else torch.maximum(mask_sum, curr)
    covered = int(mask_sum.sum().item()) if mask_sum is not None else 0
    return covered / max(1, total)


def topk_neuron_coverage(model: nn.Module, images: List[torch.Tensor], k: int) -> float:
    if len(images) == 0:
        return 0.0
    visited = set()
    total = 0
    acts0 = _forward_collect(model, images[0])
    for _, a in acts0.items():
        total += a.numel()
    for x in images:
        acts = _forward_collect(model, x)
        offset = 0
        for _, a in acts.items():
            flat = a.flatten()
            kk = min(k, flat.numel())
            idx = torch.topk(flat, kk).indices.tolist()
            for j in idx:
                visited.add(offset + j)
            offset += flat.numel()
    return len(visited) / max(1, total)


def critical_neuron_coverage(model: nn.Module, images: List[torch.Tensor], top_ratio: float = 0.1) -> float:
    if len(images) == 0:
        return 0.0
    model.eval()
    layers = _collect_named_leaf_modules(model)
    layer_offsets = {}
    total = 0
    for name, module in layers:
        layer_offsets[name] = total
        total += module.out_channels if isinstance(module, nn.Conv2d) else module.out_features

    critical = set()
    covered = set()
    for x in images:
        x = x.clone().requires_grad_(True)
        acts = {}
        handles = []
        for name, module in layers:
            def fhook(m, inp, out, n=name):
                acts[n] = out
                out.retain_grad()
            handles.append(module.register_forward_hook(fhook))
        probs = model(x).softmax(dim=1)
        pred = probs.argmax(dim=1)
        score = probs[0, pred]
        model.zero_grad(set_to_none=True)
        score.backward()
        for h in handles:
            h.remove()
        for name, module in layers:
            a = acts[name].detach()
            g = acts[name].grad.detach()
            if isinstance(module, nn.Conv2d):
                imp = (g * a).abs().mean(dim=(0, 2, 3))
                act_change = a.abs().mean(dim=(0, 2, 3))
                grad_mag = g.abs().mean(dim=(0, 2, 3))
            else:
                imp = (g * a).abs().mean(dim=0)
                act_change = a.abs().mean(dim=0)
                grad_mag = g.abs().mean(dim=0)
            k = max(1, int(len(imp) * top_ratio))
            crit_idx = torch.topk(imp, k).indices.tolist()
            grad_thr = torch.quantile(grad_mag, 0.9)
            act_thr = torch.quantile(act_change, 0.9)
            for idx in crit_idx:
                gid = layer_offsets[name] + idx
                critical.add(gid)
                if grad_mag[idx] >= grad_thr and act_change[idx] >= act_thr:
                    covered.add(gid)
    return len(covered) / max(1, len(critical))
