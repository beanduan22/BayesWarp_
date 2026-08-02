from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import random
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import torch

from bayeswarp.utils.config import load_config
from bayeswarp.utils.io import ensure_dir, save_json
from bayeswarp.utils.device import get_device
from bayeswarp.metrics.quality import SCSComputer


def spread_across_seeds(bank, count, seed):
    groups = defaultdict(list)
    for item in bank:
        groups[int(item['seed_idx'])].append(item)
    rng = random.Random(seed)
    for key in groups:
        rng.shuffle(groups[key])
    keys = sorted(groups)
    taken = []
    while len(taken) < count and keys:
        remaining = []
        for key in keys:
            if groups[key] and len(taken) < count:
                taken.append(groups[key].pop(0))
            if groups[key]:
                remaining.append(key)
        keys = remaining
    return taken


def as_batch(x):
    return x if x.ndim == 4 else x.unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--failures', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--count', type=int, default=200)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--note', default='')
    parser.add_argument('--config-label', default=None)
    parser.add_argument('--score-scs', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config)
    pack = torch.load(args.failures, map_location='cpu', weights_only=False)
    bank = pack['failure_bank']
    if not bank:
        raise RuntimeError(f'No generated cases in {args.failures}')

    selected = spread_across_seeds(bank, args.count, cfg['seed'] + args.run)

    seed_ids = sorted({int(item['seed_idx']) for item in selected})
    position = {sid: i for i, sid in enumerate(seed_ids)}
    first = {}
    for item in selected:
        first.setdefault(int(item['seed_idx']), item)

    seeds = torch.cat([as_batch(first[sid]['seed_x']) for sid in seed_ids], dim=0)
    seed_labels = torch.tensor([int(first[sid]['seed_y']) for sid in seed_ids], dtype=torch.long)
    cases = torch.cat([as_batch(item['x']) for item in selected], dim=0)
    seed_ref = torch.tensor([position[int(item['seed_idx'])] for item in selected], dtype=torch.long)
    original_pred = torch.tensor([int(item['og']) for item in selected], dtype=torch.long)
    changed_pred = torch.tensor([int(item['pred']) for item in selected], dtype=torch.long)
    target_class = torch.tensor([int(item['target_class']) for item in selected], dtype=torch.long)
    evaluation_index = torch.tensor([int(item.get('evaluation_index', -1)) for item in selected], dtype=torch.long)

    payload = {
        'seeds': seeds,
        'seed_labels': seed_labels,
        'seed_dataset_indices': torch.tensor(seed_ids, dtype=torch.long),
        'cases': cases,
        'seed_ref': seed_ref,
        'original_pred': original_pred,
        'changed_pred': changed_pred,
        'target_class': target_class,
        'evaluation_index': evaluation_index,
    }

    if args.score_scs:
        scs = SCSComputer(get_device())
        payload['scs'] = torch.tensor(
            [scs.score(as_batch(first[int(i['seed_idx'])]['seed_x']), as_batch(i['x'])) for i in selected]
        )

    out_dir = ensure_dir(args.out)
    torch.save(payload, Path(out_dir) / 'cases.pt')

    metrics = pack.get('metrics', {})
    save_json(
        {
            'config': args.config_label or Path(args.config).name,
            'dataset': cfg['dataset']['name'],
            'model': cfg['model']['name'],
            'saliency_method': cfg['bayeswarp']['saliency_method'],
            'run': args.run,
            'selection_seed': int(cfg['seed']),
            'search_seed': int(cfg['seed']) + args.run,
            'exported_cases': int(cases.size(0)),
            'unique_seeds': int(seeds.size(0)),
            'distinct_changed_classes': int(len(set(changed_pred.tolist()))),
            'source_pool_cases': int(metrics.get('NoF', len(bank))),
            'generation_num_seeds': int(cfg['test']['num_seeds']),
            'generation_budget_per_seed': int(cfg['bayeswarp']['budget']),
            'generation_max_target_classes': cfg['bayeswarp']['max_target_classes'],
            'note': args.note,
        },
        Path(out_dir) / 'manifest.json',
    )


if __name__ == '__main__':
    main()
