from __future__ import annotations
from typing import List, Optional, Sequence, Tuple

import torch

from bayeswarp.data.datasets import pixel_range, standard_augmentation
from bayeswarp.interpretability.saliency import compute_saliency
from bayeswarp.localization.region import localize_critical_region
from bayeswarp.mutation.grid_mutator import build_mutator


CONDITIONS = ('continued', 'standard', 'random', 'autoattack', 'high_scs', 'bayeswarp')


def source_samples(dataset, count: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(len(dataset), generator=generator).tolist()[:count]
    xs, ys = [], []
    for idx in order:
        x, y = dataset[idx]
        xs.append(x.unsqueeze(0))
        ys.append(int(y))
    return torch.cat(xs, dim=0), torch.tensor(ys, dtype=torch.long)


def continued_condition(X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return X.clone(), y.clone()


def standard_condition(X: torch.Tensor, y: torch.Tensor, dataset_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    transform = standard_augmentation(dataset_name)
    augmented = torch.stack([transform(X[i]) for i in range(X.size(0))], dim=0)
    return augmented, y.clone()


def random_localized_condition(
    model,
    device: torch.device,
    X: torch.Tensor,
    y: torch.Tensor,
    dataset_name: str,
    normalization: str,
    saliency_method: str,
    alpha: float,
    area_min: int,
    tau_iou: float,
    d_max: float,
    rho: float,
    n: int,
    r: float,
    eta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    p_min, p_max = pixel_range(dataset_name, normalization)
    outputs = []
    model.eval()
    for i in range(X.size(0)):
        x = X[i].unsqueeze(0).to(device)
        with torch.no_grad():
            og = int(model(x).argmax(dim=1).item())
        saliency = compute_saliency(model, x, og, saliency_method)
        mask, _ = localize_critical_region(
            saliency.detach().cpu().numpy(),
            alpha=alpha,
            area_min=area_min,
            tau_iou=tau_iou,
            d_max=d_max,
            rho=rho,
        )
        mutator = build_mutator(
            image_shape=(x.size(1), x.size(2), x.size(3)),
            region_mask=torch.from_numpy(mask).float().to(device),
            p_min=p_min,
            p_max=p_max,
            n=n,
            r=r,
            eta=eta,
        )
        u = (torch.rand(mutator.dim, device=device) * 2.0 - 1.0) * r
        outputs.append(mutator.reconstruct(u, x.squeeze(0)).detach().cpu().unsqueeze(0))
    return torch.cat(outputs, dim=0), y.clone()


def autoattack_condition(
    model,
    device: torch.device,
    X: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 8.0 / 255.0,
    batch_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        import torchattacks
    except Exception as e:
        raise ImportError(
            'The AutoAttack condition requires torchattacks. '
            'Install torchattacks to run this fine-tuning condition.'
        ) from e

    attack = torchattacks.AutoAttack(model, norm='Linf', eps=epsilon, version='standard', n_classes=int(y.max().item()) + 1)
    outputs = []
    for start in range(0, X.size(0), batch_size):
        xb = X[start:start + batch_size].to(device)
        yb = y[start:start + batch_size].to(device)
        outputs.append(attack(xb, yb).detach().cpu())
    return torch.cat(outputs, dim=0), y.clone()


def select_generated(
    failure_bank: Sequence[dict],
    count: int,
    scs_scores: Optional[Sequence[float]] = None,
    scs_threshold: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from collections import defaultdict
    import random

    items = list(failure_bank)
    if scs_threshold is not None:
        if scs_scores is None:
            raise ValueError('scs_threshold requires scs_scores')
        items = [item for item, score in zip(items, scs_scores) if score >= scs_threshold]

    groups = defaultdict(list)
    for item in items:
        groups[(item['seed_idx'], item['pred'])].append(item)
    keys = list(groups.keys())
    selected: List[dict] = []
    while len(selected) < count and keys:
        random.shuffle(keys)
        next_keys = []
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop(0))
            if groups[key]:
                next_keys.append(key)
        keys = next_keys

    if not selected:
        raise RuntimeError('No generated cases available under the requested condition.')

    X = torch.cat([item['x'] if item['x'].ndim == 4 else item['x'].unsqueeze(0) for item in selected], dim=0)
    y = torch.tensor([int(item['seed_y']) for item in selected], dtype=torch.long)
    return X, y
