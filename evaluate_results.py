from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import random
import sys
sys.path.append(str(Path(__file__).resolve().parent / 'src'))

import torch
from tqdm import tqdm

from bayeswarp.utils.config import load_config
from bayeswarp.utils.seed import set_seed
from bayeswarp.utils.io import ensure_dir, save_json
from bayeswarp.utils.device import get_device
from bayeswarp.data.datasets import dataset_meta, pixel_range
from bayeswarp.models.factory import build_model, load_checkpoint
from bayeswarp.metrics.quality import SCSComputer, compute_fid
from bayeswarp.metrics.coverage import neuron_coverage, topk_neuron_coverage, critical_neuron_coverage
from bayeswarp.finetune.sources import bank_seed_ids
from bayeswarp.pipeline import build_seeds


def default_topk(dataset_name: str) -> int:
    name = dataset_name.lower()
    if name == 'mnist':
        return 3
    if name == 'cifar10':
        return 5
    if name == 'imagenet':
        return 30
    return 5


def as_batch(x: torch.Tensor) -> torch.Tensor:
    return x if x.ndim == 4 else x.unsqueeze(0)


def fallback_indices(cfg, model, device, pack):
    seed_results = pack.get('seed_results') or []
    if seed_results and 'seed_id' in seed_results[0]:
        return None
    return [int(i) for i in build_seeds(cfg, model, device).indices]


def scs_strata(scores):
    total = max(1, len(scores))
    high = sum(1 for s in scores if s >= 0.8)
    mid = sum(1 for s in scores if 0.6 <= s < 0.8)
    low = sum(1 for s in scores if s < 0.6)
    return {
        'high': high / total,
        'mid': mid / total,
        'low': low / total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--failures', required=True)
    parser.add_argument('--label', default='main')
    parser.add_argument('--quality-samples', type=int, default=1000)
    parser.add_argument('--coverage-samples', type=int, default=256)
    parser.add_argument(
        '--scs-per-seed', type=int, default=20,
        help='Cases per source seed used for the per-seed SCS. Sampling per seed rather than '
             'globally keeps per_seed_scs defined for every seed, so paired-by-seed tests '
             'compare the same seeds across methods.',
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg['seed'])
    device = get_device()
    meta = dataset_meta(cfg['dataset']['name'])
    model = build_model(cfg['model']['name'], meta['num_classes'], pretrained=cfg['model'].get('pretrained', True)).to(device)
    load_checkpoint(model, cfg.get('checkpoint'), device, pretrained=cfg['model'].get('pretrained', True))
    model.eval()

    pack = torch.load(args.failures, map_location='cpu')
    failure_bank = pack['failure_bank']
    if len(failure_bank) == 0:
        raise RuntimeError('No generated cases to evaluate.')

    rng = random.Random(cfg['seed'])
    sample = failure_bank if len(failure_bank) <= args.quality_samples else rng.sample(failure_bank, args.quality_samples)

    real_images = [as_batch(item['seed_x']) for item in sample]
    fake_images = [as_batch(item['x']) for item in sample]
    p_min, p_max = pixel_range(cfg['dataset']['name'], cfg['dataset'].get('normalization', 'none'))

    metrics = dict(pack.get('metrics', {}))
    metrics['label'] = args.label

    seed_fallback = fallback_indices(cfg, model, device, pack)
    if 'seed_ids' not in metrics:
        seed_results = pack.get('seed_results') or []
        if seed_results and 'seed_id' in seed_results[0]:
            metrics['seed_ids'] = [int(r['seed_id']) for r in seed_results]
        elif seed_fallback is not None and len(seed_fallback) == len(seed_results):
            metrics['seed_ids'] = list(seed_fallback)

    try:
        metrics['FID'] = compute_fid(real_images, fake_images, device, p_min, p_max)
    except Exception as e:
        metrics['FID'] = f'Unavailable: {e}'

    try:
        scs = SCSComputer(device, p_min, p_max)
        scores = [
            scs.score(real, fake)
            for real, fake in tqdm(list(zip(real_images, fake_images)), desc='SCS')
        ]
        metrics['SCS'] = float(sum(scores) / max(1, len(scores)))
        metrics['SCS_strata'] = scs_strata(scores)

        bank_ids = bank_seed_ids(pack, seed_fallback)
        by_seed = defaultdict(list)
        for seed_id, item in zip(bank_ids, failure_bank):
            by_seed[int(seed_id)].append(item)
        strat_rng = random.Random(cfg['seed'])
        scs_seed_ids, per_seed_scs, per_seed_counts = [], [], []
        for seed_id in tqdm(sorted(by_seed), desc='SCS per seed'):
            items = by_seed[seed_id]
            if len(items) > args.scs_per_seed:
                items = strat_rng.sample(items, args.scs_per_seed)
            seed_scores = [scs.score(as_batch(it['seed_x']), as_batch(it['x'])) for it in items]
            scs_seed_ids.append(int(seed_id))
            per_seed_scs.append(float(sum(seed_scores) / len(seed_scores)))
            per_seed_counts.append(len(seed_scores))
        metrics['scs_seed_ids'] = scs_seed_ids
        metrics['per_seed_scs'] = per_seed_scs
        metrics['per_seed_scs_cases'] = per_seed_counts
        metrics['scs_per_seed_cap'] = int(args.scs_per_seed)
        all_ids = [int(v) for v in metrics.get('seed_ids', [])]
        metrics['scs_seeds_without_cases'] = sorted(set(all_ids) - set(scs_seed_ids))
    except Exception as e:
        metrics['SCS'] = f'Unavailable: {e}'

    cov_images = [x.to(device) for x in fake_images[: args.coverage_samples]]
    cov_seeds = [x.to(device) for x in real_images[: args.coverage_samples]]
    metrics['NC'] = neuron_coverage(model, cov_images)
    metrics['TKNC'] = topk_neuron_coverage(model, cov_images, k=default_topk(cfg['dataset']['name']))
    metrics['CNC'] = critical_neuron_coverage(model, cov_images, cov_seeds)

    out_dir = ensure_dir(cfg['output_dir'])
    save_json(metrics, out_dir / f'evaluation_{args.label}.json')


if __name__ == '__main__':
    main()
