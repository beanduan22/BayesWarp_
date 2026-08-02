from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import torch
from tqdm import tqdm

from bayeswarp.data.datasets import (
    build_seed_dataset,
    dataset_meta,
    pixel_range,
    select_correctly_classified_seeds,
)
from bayeswarp.metrics.failure import compute_failure_metrics
from bayeswarp.models.factory import build_model, load_checkpoint
from bayeswarp.testing.bayeswarp import BayesWarpConfig, BayesWarpTester
from bayeswarp.utils.seed import set_seed


def build_subject(cfg: Dict[str, Any], device: torch.device):
    meta = dataset_meta(cfg['dataset']['name'])
    model = build_model(
        cfg['model']['name'],
        meta['num_classes'],
        pretrained=cfg['model'].get('pretrained', True),
    ).to(device)
    load_checkpoint(model, cfg.get('checkpoint'), device, pretrained=cfg['model'].get('pretrained', True))
    model.eval()
    return model


def build_seeds(cfg: Dict[str, Any], model, device: torch.device, skip: int = 0, num_seeds: Optional[int] = None):
    seed_ds = build_seed_dataset(
        cfg['dataset']['name'],
        cfg['dataset']['root'],
        normalization=cfg['dataset'].get('normalization', 'none'),
        image_size=cfg['dataset'].get('image_size'),
        split=cfg['dataset'].get('seed_split', 'train'),
    )
    return select_correctly_classified_seeds(
        model,
        seed_ds,
        device,
        num_seeds=num_seeds if num_seeds is not None else cfg['test']['num_seeds'],
        seed=cfg['seed'],
        skip=skip,
    )


def generate(
    cfg: Dict[str, Any],
    device: torch.device,
    ablation: str = 'none',
    run: int = 0,
    skip: int = 0,
    num_seeds: Optional[int] = None,
    progress: str = 'Generating test cases',
) -> Tuple[list, list, Dict[str, Any]]:
    set_seed(cfg['seed'])
    model = build_subject(cfg, device)
    seed_subset = build_seeds(cfg, model, device, skip=skip, num_seeds=num_seeds)

    set_seed(cfg['seed'] + run)
    p_min, p_max = pixel_range(cfg['dataset']['name'], cfg['dataset'].get('normalization', 'none'))
    bw_cfg = BayesWarpConfig(ablation=ablation, **cfg['bayeswarp'])
    tester = BayesWarpTester(model, device, bw_cfg, p_min, p_max)

    seed_results = []
    failure_bank = []
    for i in tqdm(range(len(seed_subset)), desc=progress):
        x, y = seed_subset[i]
        result = tester.run_on_seed(x)
        result['seed_idx'] = i
        result['seed_y'] = int(y)
        seed_results.append(result)
        for f in result['failures']:
            failure_bank.append({'seed_idx': i, 'seed_x': x.cpu(), 'seed_y': int(y), **f})

    metrics = compute_failure_metrics(seed_results, budget=bw_cfg.budget)
    metrics['method'] = 'bayeswarp'
    metrics['ablation'] = ablation
    metrics['run'] = run
    return seed_results, failure_bank, metrics
