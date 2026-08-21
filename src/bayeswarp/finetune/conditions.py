"""Fine-tuning conditions for the controlled study.

Protocol (identical for every condition):

* the same ``num_sources`` source samples, identified by dataset index in the
  manifest built by ``scripts/build_finetune_sources.py``;
* exactly ``per_source`` added training samples derived from each source;
* source order preserved, so added sample ``k`` of source ``i`` always sits at
  position ``i * per_source + k``.

Only the derivation differs between conditions. Each condition therefore returns
``(X, y, info)`` where ``info`` records condition-specific diagnostics, most
importantly how many *distinct* samples the derivation actually produced per
source: ``continued`` duplicates by construction, and AutoAttack is close to
deterministic, so those numbers belong in the results rather than in a footnote.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from bayeswarp.data.datasets import pixel_range, standard_augmentation
from bayeswarp.finetune.sources import expand_labels
from bayeswarp.interpretability.saliency import compute_saliency
from bayeswarp.localization.region import localize_critical_region
from bayeswarp.mutation.grid_mutator import build_mutator


CONDITIONS = ('continued', 'standard', 'random', 'autoattack', 'high_scs', 'bayeswarp')

GENERATED_CONDITIONS = ('high_scs', 'bayeswarp')


def _check_shape(X: torch.Tensor, num_sources: int, per_source: int) -> None:
    expected = num_sources * per_source
    if X.size(0) != expected:
        raise RuntimeError(
            f'Condition produced {X.size(0)} added samples, expected {expected} '
            f'({num_sources} sources x {per_source} per source).'
        )


def _distinct_per_source(X: torch.Tensor, num_sources: int, per_source: int) -> float:
    """Mean number of pairwise-distinct added samples per source."""
    flat = X.reshape(num_sources, per_source, -1)
    totals = 0
    for i in range(num_sources):
        block = flat[i]
        keep = 1
        for k in range(1, per_source):
            if not any(torch.equal(block[k], block[j]) for j in range(k)):
                keep += 1
        totals += keep
    return float(totals / max(1, num_sources))


def continued_condition(
    X: torch.Tensor,
    y: torch.Tensor,
    per_source: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Unmodified source samples, repeated to match the added-sample budget.

    The repetition is deliberate: this condition holds the number of added
    samples and the number of gradient steps fixed while contributing no new
    information, which is what the other conditions are measured against.
    """
    added = X.repeat_interleave(int(per_source), dim=0)
    _check_shape(added, X.size(0), per_source)
    return added, expand_labels(y, per_source), {
        'derivation': 'repeat',
        'distinct_per_source': 1.0,
        'note': 'Identical copies by construction: a budget-matched no-new-information control.',
    }


def standard_condition(
    X: torch.Tensor,
    y: torch.Tensor,
    dataset_name: str,
    per_source: int,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """``per_source`` independent draws from the dataset's standard augmentation."""
    transform = standard_augmentation(dataset_name)
    outputs: List[torch.Tensor] = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        for i in range(X.size(0)):
            for _ in range(int(per_source)):
                outputs.append(transform(X[i]).unsqueeze(0))
    added = torch.cat(outputs, dim=0)
    _check_shape(added, X.size(0), per_source)
    info = {
        'derivation': 'standard_augmentation',
        'distinct_per_source': _distinct_per_source(added, X.size(0), per_source),
    }
    if info['distinct_per_source'] < 1.5:
        info['note'] = (
            'This dataset has a near-identity standard augmentation, so this '
            'condition largely coincides with the continued-training control.'
        )
    return added, expand_labels(y, per_source), info


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
    per_source: int,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """``per_source`` independent uniform perturbations inside the same critical region.

    The region and the mutator are built once per source, so the only thing that
    varies across the ``per_source`` draws is the perturbation vector: this is
    BayesWarp's search space explored at random instead of by the surrogate.
    """
    p_min, p_max = pixel_range(dataset_name, normalization)
    generator = torch.Generator().manual_seed(int(seed))
    outputs: List[torch.Tensor] = []
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
        for _ in range(int(per_source)):
            u = (torch.rand(mutator.dim, generator=generator) * 2.0 - 1.0) * r
            u = u.to(device)
            outputs.append(mutator.reconstruct(u, x.squeeze(0)).detach().cpu().unsqueeze(0))
    added = torch.cat(outputs, dim=0)
    _check_shape(added, X.size(0), per_source)
    return added, expand_labels(y, per_source), {
        'derivation': 'random_localized',
        'distinct_per_source': _distinct_per_source(added, X.size(0), per_source),
    }


class _UnitRangeModel(nn.Module):
    def __init__(self, model: nn.Module, p_min: torch.Tensor, p_max: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer('lo', p_min.view(1, -1, 1, 1).clone())
        self.register_buffer('hi', p_max.view(1, -1, 1, 1).clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.lo + x * (self.hi - self.lo))


def autoattack_condition(
    model,
    device: torch.device,
    X: torch.Tensor,
    y: torch.Tensor,
    dataset_name: str,
    normalization: str,
    per_source: int,
    epsilon: float = 8.0 / 255.0,
    batch_size: int = 32,
    seed: int = 0,
    num_classes: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """``per_source`` AutoAttack rounds per source, each with a different attack seed.

    AutoAttack is only partly randomised, so repeated rounds can collapse onto
    the same adversarial example. ``distinct_per_source`` in the returned info
    reports how often that happened instead of hiding it.
    """
    try:
        import torchattacks
    except Exception as e:
        raise ImportError(
            'The AutoAttack condition requires torchattacks. '
            'Install torchattacks to run this fine-tuning condition.'
        ) from e

    p_min, p_max = pixel_range(dataset_name, normalization)
    lo = p_min.to(device).view(1, -1, 1, 1)
    hi = p_max.to(device).view(1, -1, 1, 1)
    wrapped = _UnitRangeModel(model, p_min.to(device), p_max.to(device)).to(device).eval()
    classes = int(num_classes) if num_classes is not None else int(y.max().item()) + 1

    rounds: List[torch.Tensor] = []
    for repetition in range(int(per_source)):
        attack_seed = int(seed) + repetition
        try:
            attack = torchattacks.AutoAttack(
                wrapped, norm='Linf', eps=epsilon, version='standard',
                n_classes=classes, seed=attack_seed,
            )
        except TypeError:
            attack = torchattacks.AutoAttack(
                wrapped, norm='Linf', eps=epsilon, version='standard', n_classes=classes,
            )
        chunks: List[torch.Tensor] = []
        for start in range(0, X.size(0), batch_size):
            xb = X[start:start + batch_size].to(device)
            yb = y[start:start + batch_size].to(device)
            unit = ((xb - lo) / (hi - lo)).clamp(0, 1)
            adv = attack(unit, yb).detach()
            chunks.append((lo + adv * (hi - lo)).cpu())
        rounds.append(torch.cat(chunks, dim=0))

    # rounds[k][i] -> position i * per_source + k
    added = torch.stack(rounds, dim=1).reshape(-1, *X.shape[1:])
    _check_shape(added, X.size(0), per_source)
    info = {
        'derivation': 'autoattack',
        'epsilon': float(epsilon),
        'distinct_per_source': _distinct_per_source(added, X.size(0), per_source),
    }
    if info['distinct_per_source'] < 1.5:
        info['note'] = (
            'AutoAttack converged to (near-)identical solutions across attack seeds, '
            'so this condition contributes close to one distinct sample per source.'
        )
    return added, expand_labels(y, per_source), info


def group_bank_by_source(
    failure_bank: Sequence[dict],
    seed_ids: Sequence[int],
    scs_scores: Optional[Sequence[float]] = None,
    scs_threshold: Optional[float] = None,
) -> Dict[int, List[dict]]:
    """Group generated cases by source dataset index, optionally filtered by SCS."""
    if len(seed_ids) != len(failure_bank):
        raise ValueError('seed_ids must have one entry per case in the failure bank')
    if scs_threshold is not None and scs_scores is None:
        raise ValueError('scs_threshold requires scs_scores')
    if scs_scores is not None and len(scs_scores) != len(failure_bank):
        raise ValueError('scs_scores must have one entry per case in the failure bank')

    groups: Dict[int, List[dict]] = defaultdict(list)
    for position, (item, seed_id) in enumerate(zip(failure_bank, seed_ids)):
        if scs_threshold is not None and float(scs_scores[position]) < float(scs_threshold):
            continue
        groups[int(seed_id)].append(item)
    return dict(groups)


def select_generated(
    groups: Dict[int, List[dict]],
    source_ids: Sequence[int],
    per_source: int,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Take exactly ``per_source`` generated cases from each requested source.

    Within a source, cases are drawn round-robin over the predicted class so the
    selection is not dominated by whichever target class happened to be easiest;
    the order inside each class is shuffled with the supplied seed.
    """
    import random

    rng = random.Random(int(seed))
    images: List[torch.Tensor] = []
    labels: List[int] = []
    class_counts: Dict[int, int] = {}
    missing: List[Tuple[int, int]] = []

    for source_id in source_ids:
        available = list(groups.get(int(source_id), []))
        if len(available) < int(per_source):
            missing.append((int(source_id), len(available)))
            continue
        by_class: Dict[int, List[dict]] = defaultdict(list)
        for item in available:
            by_class[int(item['pred'])].append(item)
        for bucket in by_class.values():
            rng.shuffle(bucket)
        keys = sorted(by_class)
        rng.shuffle(keys)

        chosen: List[dict] = []
        while len(chosen) < int(per_source):
            progressed = False
            for key in keys:
                if not by_class[key]:
                    continue
                chosen.append(by_class[key].pop(0))
                progressed = True
                if len(chosen) >= int(per_source):
                    break
            if not progressed:
                break
        for item in chosen:
            x = item['x'] if item['x'].ndim == 4 else item['x'].unsqueeze(0)
            images.append(x)
            labels.append(int(item['seed_y']))
            class_counts[int(item['pred'])] = class_counts.get(int(item['pred']), 0) + 1

    if missing:
        preview = ', '.join(f'seed {sid}: {count}' for sid, count in missing[:5])
        raise RuntimeError(
            f'{len(missing)} of {len(source_ids)} sources have fewer than {per_source} '
            f'generated cases under this condition ({preview}). '
            'Rebuild the source manifest against this failure pack so every condition '
            'can supply the same number of samples per source.'
        )

    added = torch.cat(images, dim=0)
    _check_shape(added, len(source_ids), per_source)
    return added, torch.tensor(labels, dtype=torch.long), {
        'derivation': 'generated',
        'distinct_per_source': _distinct_per_source(added, len(source_ids), per_source),
        'target_classes_used': len(class_counts),
    }
