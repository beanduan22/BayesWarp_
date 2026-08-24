from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import random
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import torch

from bayeswarp.utils.config import load_config
from bayeswarp.utils.io import ensure_dir
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
    parser.add_argument('--score-scs', action='store_true')
    parser.add_argument('--images', action='store_true', help='Also write browsable PNG pairs.')
    parser.add_argument('--image-limit', type=int, default=40, help='How many PNG pairs to write.')
    args = parser.parse_args()

    cfg = load_config(args.config)
    pack = torch.load(args.failures, map_location='cpu', weights_only=False)
    bank = pack['failure_bank']
    if not bank:
        raise RuntimeError(f'No generated cases in {args.failures}')

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

    if args.images:
        from torchvision.utils import save_image
        image_dir = ensure_dir(Path(out_dir) / 'images')
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

    print(f'{args.out}: {int(cases_u8.size(0))} cases from {int(seeds_u8.size(0))} seeds')


if __name__ == '__main__':
    main()
