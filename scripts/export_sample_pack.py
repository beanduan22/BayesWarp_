"""Export a representative sample pack for the reproduction package.

The pack is a browsable, size-bounded view of a real generation run:

* ``images/`` holds PNG pairs so a reviewer can look at the cases on GitHub
  without downloading or running anything;
* ``cases.pt`` holds the same cases as uint8 tensors plus their metadata,
  roughly 20-40x smaller than the float32 originals;
* ``manifest.json`` records what the run *actually* used, read back from the
  failure pack rather than from the config file, so the manifest cannot drift
  away from the artefact it describes.
"""
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
from bayeswarp.data.datasets import pixel_range
from bayeswarp.metrics.quality import SCSComputer, to_unit_range
from bayeswarp.finetune.sources import bank_seed_ids


def as_batch(x):
    return x if x.ndim == 4 else x.unsqueeze(0)


def to_uint8(x: torch.Tensor, p_min: torch.Tensor, p_max: torch.Tensor) -> torch.Tensor:
    unit = to_unit_range(as_batch(x).detach().float(), p_min, p_max)
    return (unit.clamp(0, 1) * 255.0).round().to(torch.uint8)


def spread_across_seeds(bank, seed_ids, count, seed):
    """Round-robin over source seeds so the export is not one seed's output."""
    groups = defaultdict(list)
    for item, seed_id in zip(bank, seed_ids):
        groups[int(seed_id)].append(item)
    rng = random.Random(seed)
    for key in groups:
        rng.shuffle(groups[key])
    keys = sorted(groups)
    taken = []
    while len(taken) < count and keys:
        remaining = []
        for key in keys:
            if groups[key] and len(taken) < count:
                taken.append((key, groups[key].pop(0)))
            if groups[key]:
                remaining.append(key)
        keys = remaining
    return taken


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
    parser.add_argument('--images', action='store_true', help='Also write browsable PNG pairs.')
    parser.add_argument('--image-limit', type=int, default=40, help='How many PNG pairs to write.')
    args = parser.parse_args()

    cfg = load_config(args.config)
    pack = torch.load(args.failures, map_location='cpu', weights_only=False)
    bank = pack['failure_bank']
    if not bank:
        raise RuntimeError(f'No generated cases in {args.failures}')

    metrics = pack.get('metrics', {})
    seed_results = pack.get('seed_results') or []
    seed_ids = bank_seed_ids(pack, None) if (seed_results and 'seed_id' in seed_results[0]) \
        else [int(item['seed_idx']) for item in bank]

    selected = spread_across_seeds(bank, seed_ids, args.count, cfg['seed'] + args.run)
    if not selected:
        raise RuntimeError('Nothing selected for export')

    unique_ids = sorted({sid for sid, _ in selected})
    position = {sid: i for i, sid in enumerate(unique_ids)}
    first = {}
    for sid, item in selected:
        first.setdefault(sid, item)

    p_min, p_max = pixel_range(cfg['dataset']['name'], cfg['dataset'].get('normalization', 'none'))

    seeds_u8 = torch.cat([to_uint8(first[sid]['seed_x'], p_min, p_max) for sid in unique_ids], dim=0)
    cases_u8 = torch.cat([to_uint8(item['x'], p_min, p_max) for _, item in selected], dim=0)

    payload = {
        'format': 'uint8',
        'value_range': [0, 255],
        'note': 'Images are stored in display space; map back with p_min/p_max from the config.',
        'seeds': seeds_u8,
        'seed_labels': torch.tensor([int(first[sid]['seed_y']) for sid in unique_ids], dtype=torch.long),
        'seed_dataset_indices': torch.tensor(unique_ids, dtype=torch.long),
        'cases': cases_u8,
        'seed_ref': torch.tensor([position[sid] for sid, _ in selected], dtype=torch.long),
        'original_pred': torch.tensor([int(i['og']) for _, i in selected], dtype=torch.long),
        'changed_pred': torch.tensor([int(i['pred']) for _, i in selected], dtype=torch.long),
        'target_class': torch.tensor([int(i['target_class']) for _, i in selected], dtype=torch.long),
        'evaluation_index': torch.tensor(
            [int(i.get('evaluation_index', -1)) for _, i in selected], dtype=torch.long
        ),
    }

    scs_values = None
    if args.score_scs:
        scs = SCSComputer(get_device(), p_min, p_max)
        scs_values = [
            float(scs.score(as_batch(first[sid]['seed_x']), as_batch(item['x'])))
            for sid, item in selected
        ]
        payload['scs'] = torch.tensor(scs_values)

    out_dir = ensure_dir(args.out)
    torch.save(payload, Path(out_dir) / 'cases.pt')

    image_records = []
    if args.images:
        from torchvision.utils import save_image
        image_dir = ensure_dir(Path(out_dir) / 'images')
        # Best case per seed first, then round-robin, so the PNG set shows
        # different source images rather than one seed's top-N.
        by_seed = defaultdict(list)
        for index, (sid, _) in enumerate(selected):
            by_seed[sid].append(index)
        for sid in by_seed:
            by_seed[sid].sort(key=lambda i: -scs_values[i] if scs_values else i)
        order, keys = [], sorted(by_seed)
        while keys:
            remaining = []
            for sid in keys:
                order.append(by_seed[sid].pop(0))
                if by_seed[sid]:
                    remaining.append(sid)
            keys = remaining
        for rank, index in enumerate(order[: args.image_limit]):
            sid, item = selected[index]
            stem = f'{rank:03d}_seed{sid}_{int(item["og"])}to{int(item["pred"])}'
            if scs_values:
                stem += f'_scs{scs_values[index]:.3f}'
            save_image(to_uint8(first[sid]['seed_x'], p_min, p_max).float() / 255.0,
                       str(Path(image_dir) / f'{stem}_original.png'))
            save_image(to_uint8(item['x'], p_min, p_max).float() / 255.0,
                       str(Path(image_dir) / f'{stem}_generated.png'))
            image_records.append({
                'file_stem': stem,
                'seed_id': int(sid),
                'source_label': int(item['seed_y']),
                'original_pred': int(item['og']),
                'generated_pred': int(item['pred']),
                'scs': round(scs_values[index], 4) if scs_values else None,
            })

    # Everything below is read back from the pack, not from the config, so the
    # manifest describes the run that produced these cases.
    generation = {
        'seeds_run': len(seed_results) or None,
        'budget_per_seed': int(seed_results[0]['evaluations_used']) if seed_results else None,
        'mutation_dim': int(seed_results[0]['mutation_dim']) if seed_results and 'mutation_dim' in seed_results[0] else None,
        'store_cases_per_seed': metrics.get('store_cases_per_seed'),
        'NoF': metrics.get('NoF'),
        'FSR': metrics.get('FSR'),
        'DoF': metrics.get('DoF'),
        'stored_cases': metrics.get('stored_cases', len(bank)),
    }
    manifest = {
        'config': args.config_label or Path(args.config).name,
        'dataset': cfg['dataset']['name'],
        'model': cfg['model']['name'],
        'saliency_method': cfg['bayeswarp']['saliency_method'],
        'seed_split': cfg['dataset'].get('seed_split', 'train'),
        'run': args.run,
        'selection_seed': int(cfg['seed']),
        'search_seed': int(cfg['seed']) + args.run,
        'exported_cases': int(cases_u8.size(0)),
        'unique_seeds': int(seeds_u8.size(0)),
        'distinct_changed_classes': int(len(set(payload['changed_pred'].tolist()))),
        'source_pool_cases': int(len(bank)),
        'storage': 'uint8 tensors in cases.pt; PNG pairs in images/',
        'generation': generation,
        'configured': {
            'num_seeds': int(cfg['test']['num_seeds']),
            'budget': int(cfg['bayeswarp']['budget']),
            'max_target_classes': cfg['bayeswarp']['max_target_classes'],
            'r': cfg['bayeswarp']['r'],
            'eta': cfg['bayeswarp']['eta'],
            'n': cfg['bayeswarp']['n'],
        },
        'note': args.note,
    }
    if scs_values:
        ordered = sorted(scs_values)
        manifest['scs'] = {
            'mean': round(sum(scs_values) / len(scs_values), 4),
            'median': round(ordered[len(ordered) // 2], 4),
            'min': round(ordered[0], 4),
            'max': round(ordered[-1], 4),
            'share_at_or_above_0.8': round(sum(1 for v in scs_values if v >= 0.8) / len(scs_values), 4),
        }
    if image_records:
        manifest['images'] = image_records
    save_json(manifest, Path(out_dir) / 'manifest.json')
    print(f'{args.out}: {manifest["exported_cases"]} cases from {manifest["unique_seeds"]} seeds')


if __name__ == '__main__':
    main()
