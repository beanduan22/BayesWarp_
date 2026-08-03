from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
import torch
import torch.nn as nn


SUPPORTED_LEAF_TYPES = (nn.Conv2d, nn.Linear)
DEFAULT_NC_THRESHOLD = 0.25
CRITICAL_TOP_RATIO = 0.1
CRITICAL_QUANTILE = 0.9


def _collect_named_leaf_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    modules = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0 and isinstance(module, SUPPORTED_LEAF_TYPES):
            modules.append((name, module))
    return modules


def _neuron_count(module: nn.Module) -> int:
    return module.out_channels if isinstance(module, nn.Conv2d) else module.out_features


def _neuron_activations(output: torch.Tensor) -> torch.Tensor:
    if output.ndim == 4:
        return output.mean(dim=(0, 2, 3))
    return output.mean(dim=0)


def _forward_collect(model: nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    acts = {}
    handles = []
    for name, module in _collect_named_leaf_modules(model):
        handles.append(
            module.register_forward_hook(
                lambda m, i, o, n=name: acts.__setitem__(n, _neuron_activations(o.detach()))
            )
        )
    _ = model(x)
    for h in handles:
        h.remove()
    return acts


def _scale_to_unit(values: torch.Tensor) -> torch.Tensor:
    low = values.min()
    high = values.max()
    return (values - low) / (high - low + 1e-8)


def _layer_offsets(layers: Sequence[Tuple[str, nn.Module]]) -> Tuple[Dict[str, int], int]:
    offsets = {}
    total = 0
    for name, module in layers:
        offsets[name] = total
        total += _neuron_count(module)
    return offsets, total


def neuron_coverage(
    model: nn.Module,
    images: List[torch.Tensor],
    threshold: float = DEFAULT_NC_THRESHOLD,
) -> float:
    if len(images) == 0:
        return 0.0
    layers = _collect_named_leaf_modules(model)
    offsets, total = _layer_offsets(layers)
    covered = set()
    for x in images:
        acts = _forward_collect(model, x)
        for name, _ in layers:
            scaled = _scale_to_unit(acts[name])
            for idx in torch.nonzero(scaled > threshold, as_tuple=False).flatten().tolist():
                covered.add(offsets[name] + idx)
    return len(covered) / max(1, total)


def topk_neuron_coverage(model: nn.Module, images: List[torch.Tensor], k: int) -> float:
    if len(images) == 0:
        return 0.0
    layers = _collect_named_leaf_modules(model)
    offsets, total = _layer_offsets(layers)
    visited = set()
    for x in images:
        acts = _forward_collect(model, x)
        for name, _ in layers:
            flat = acts[name]
            kk = min(k, flat.numel())
            for idx in torch.topk(flat, kk).indices.tolist():
                visited.add(offsets[name] + idx)
    return len(visited) / max(1, total)


def critical_neuron_coverage(
    model: nn.Module,
    images: List[torch.Tensor],
    seeds: List[torch.Tensor],
    top_ratio: float = CRITICAL_TOP_RATIO,
) -> float:
    if len(images) == 0:
        return 0.0
    if len(seeds) != len(images):
        raise ValueError('critical_neuron_coverage requires one seed image per generated image.')
    model.eval()
    layers = _collect_named_leaf_modules(model)
    offsets, _ = _layer_offsets(layers)

    critical = set()
    covered = set()
    for x, seed in zip(images, seeds):
        with torch.no_grad():
            seed_acts = _forward_collect(model, seed.to(x.device))

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

        for name, _ in layers:
            a = _neuron_activations(acts[name].detach())
            g = _neuron_activations(acts[name].grad.detach().abs())
            imp = (g * a).abs()
            act_change = (a - seed_acts[name]).abs()
            k = max(1, int(imp.numel() * top_ratio))
            grad_thr = torch.quantile(g, CRITICAL_QUANTILE)
            act_thr = torch.quantile(act_change, CRITICAL_QUANTILE)
            for idx in torch.topk(imp, k).indices.tolist():
                gid = offsets[name] + idx
                critical.add(gid)
                if g[idx] >= grad_thr and act_change[idx] >= act_thr:
                    covered.add(gid)
    return len(covered) / max(1, len(critical))
