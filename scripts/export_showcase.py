"""Export a small set of high-fidelity generated cases per dataset.

These are the figure-quality examples: cases whose SCS says the mutant still
reads as the original class to a human, while the model's prediction changed.
One case per source seed, spread over source classes, so the showcase is not
ten variations of the same image.

Each case is written as an original/generated PNG pair plus a side-by-side
contact sheet, with a JSON record naming the source seed, the label, and both
predictions.
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
from bayeswarp.utils.device import get_device
from bayeswarp.utils.io import ensure_dir, save_json
from bayeswarp.data.datasets import pixel_range
from bayeswarp.metrics.quality import SCSComputer, to_unit_range
from bayeswarp.finetune.sources import bank_seed_ids


def as_batch(x):
    return x if x.ndim == 4 else x.unsqueeze(0)


def to_image(x, p_min, p_max):
    return to_unit_range(as_batch(x).detach().float(), p_min, p_max).squeeze(0).clamp(0, 1)


CIFAR10_CLASSES = (
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
)


def builtin_class_names(dataset: str):
    """Class names for the shipped datasets, so records read as words not ids."""
    name = dataset.lower()
    if name == 'mnist':
        return {str(i): f'digit {i}' for i in range(10)}
    if name == 'cifar10':
        return {str(i): label for i, label in enumerate(CIFAR10_CLASSES)}
    if name == 'imagenet':
        try:
            from torchvision.models import ResNet50_Weights
            categories = ResNet50_Weights.IMAGENET1K_V1.meta['categories']
            return {str(i): label for i, label in enumerate(categories)}
        except Exception:
            return {}
    return {}


def load_class_names(path, dataset: str):
    if not path:
        return builtin_class_names(dataset)
    import json
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    return {str(k): (v.get('name') if isinstance(v, dict) else str(v)) for k, v in payload.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--failures', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument(
        '--candidates', type=int, default=3000,
        help='Cases to score with SCS before ranking. Scoring the whole bank is wasteful.',
    )
    parser.add_argument('--min-scs', type=float, default=0.8)
    parser.add_argument('--class-names', default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    p_min, p_max = pixel_range(cfg['dataset']['name'], cfg['dataset'].get('normalization', 'none'))
    pack = torch.load(args.failures, map_location='cpu', weights_only=False)
    bank = pack['failure_bank']
    if not bank:
        raise RuntimeError(f'No generated cases in {args.failures}')

    seed_results = pack.get('seed_results') or []
    seed_ids = bank_seed_ids(pack, None) if (seed_results and 'seed_id' in seed_results[0]) \
        else [int(item['seed_idx']) for item in bank]

    # Sample candidates evenly across seeds before scoring, so the SCS budget is
    # not spent on whichever seed happened to produce the most cases.
    grouped = defaultdict(list)
    for item, seed_id in zip(bank, seed_ids):
        grouped[int(seed_id)].append(item)
    rng = random.Random(args.seed)
    for key in grouped:
        rng.shuffle(grouped[key])
    keys = sorted(grouped)
    candidates = []
    while len(candidates) < args.candidates and keys:
        remaining = []
        for key in keys:
            if grouped[key] and len(candidates) < args.candidates:
                candidates.append((key, grouped[key].pop(0)))
            if grouped[key]:
                remaining.append(key)
        keys = remaining

    scs = SCSComputer(get_device(), p_min, p_max)
    scored = []
    for seed_id, item in candidates:
        value = float(scs.score(as_batch(item['seed_x']), as_batch(item['x'])))
        scored.append({'seed_id': seed_id, 'item': item, 'scs': value})

    eligible = [s for s in scored if s['scs'] >= args.min_scs]
    if not eligible:
        eligible = scored
    eligible.sort(key=lambda s: -s['scs'])

    # One case per seed, rotating over source classes.
    picked, used_seeds, class_counts = [], set(), defaultdict(int)
    for entry in sorted(eligible, key=lambda s: (class_counts[int(s['item']['seed_y'])], -s['scs'])):
        if len(picked) >= args.count:
            break
        if entry['seed_id'] in used_seeds:
            continue
        picked.append(entry)
        used_seeds.add(entry['seed_id'])
        class_counts[int(entry['item']['seed_y'])] += 1
    for entry in eligible:
        if len(picked) >= args.count:
            break
        if entry['seed_id'] not in used_seeds:
            picked.append(entry)
            used_seeds.add(entry['seed_id'])

    from torchvision.utils import make_grid, save_image
    out_dir = ensure_dir(args.out)
    names = load_class_names(args.class_names, cfg['dataset']['name'])

    records, tiles = [], []
    for rank, entry in enumerate(picked):
        item = entry['item']
        original = to_image(item['seed_x'], p_min, p_max)
        generated = to_image(item['x'], p_min, p_max)
        stem = f"{rank:02d}_seed{entry['seed_id']}_{int(item['og'])}to{int(item['pred'])}_scs{entry['scs']:.3f}"
        save_image(original, str(Path(out_dir) / f'{stem}_original.png'))
        save_image(generated, str(Path(out_dir) / f'{stem}_generated.png'))
        tiles.extend([
            original.repeat(3, 1, 1) if original.size(0) == 1 else original,
            generated.repeat(3, 1, 1) if generated.size(0) == 1 else generated,
        ])
        records.append({
            'file_stem': stem,
            'seed_id': int(entry['seed_id']),
            'source_label': int(item['seed_y']),
            'source_class_name': names.get(str(int(item['seed_y']))),
            'original_pred': int(item['og']),
            'generated_pred': int(item['pred']),
            'generated_class_name': names.get(str(int(item['pred']))),
            'scs': round(entry['scs'], 4),
        })

    if tiles:
        save_image(
            make_grid(torch.stack(tiles), nrow=2, padding=2, pad_value=1.0),
            str(Path(out_dir) / 'contact_sheet.png'),
        )

    save_json(
        {
            'config': Path(args.config).name,
            'dataset': cfg['dataset']['name'],
            'model': cfg['model']['name'],
            'source': args.failures,
            'selection': (
                f'top SCS with at least {args.min_scs}, one case per source seed, '
                'rotated over source classes'
            ),
            'candidates_scored': len(scored),
            'candidates_at_or_above_min_scs': len([s for s in scored if s['scs'] >= args.min_scs]),
            'exported': len(records),
            'distinct_seeds': len(used_seeds),
            'distinct_source_classes': len({r['source_label'] for r in records}),
            'scs_min': min((r['scs'] for r in records), default=None),
            'scs_max': max((r['scs'] for r in records), default=None),
            'cases': records,
        },
        Path(out_dir) / 'showcase.json',
    )
    print(f'{args.out}: {len(records)} cases, SCS '
          f'{min((r["scs"] for r in records), default=0):.3f}-{max((r["scs"] for r in records), default=0):.3f}')


if __name__ == '__main__':
    main()
