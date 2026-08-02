from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import torch

from bayeswarp.utils.device import get_device
from bayeswarp.utils.io import ensure_dir
from bayeswarp.metrics.quality import SCSComputer

STRATA = (('high', 0.8, 1.01), ('mid', 0.6, 0.8), ('low', -1.01, 0.6))


def to_image(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        x = x.squeeze(0)
    x = x.detach().float()
    lo, hi = float(x.min()), float(x.max())
    if hi - lo > 1e-8:
        x = (x - lo) / (hi - lo)
    return x.clamp(0, 1)


def save_png(tensor: torch.Tensor, path: Path) -> None:
    from torchvision.utils import save_image
    save_image(tensor, str(path))


def build_pair_grid(pairs, path: Path) -> None:
    from torchvision.utils import make_grid, save_image
    if not pairs:
        return
    tiles = []
    for seed, gen in pairs:
        seed, gen = to_image(seed), to_image(gen)
        if seed.size(0) == 1:
            seed, gen = seed.repeat(3, 1, 1), gen.repeat(3, 1, 1)
        tiles.extend([seed, gen])
    grid = make_grid(torch.stack(tiles), nrow=2, padding=2, pad_value=1.0)
    save_image(grid, str(path))


def spread_across_seeds(members, count):
    groups = {}
    for item in members:
        groups.setdefault(item['seed_idx'], []).append(item)
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


def main():
    parser = argparse.ArgumentParser(
        description='Stratify generated cases by per-case SCS and export originals with their mutants.'
    )
    parser.add_argument('--failures', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--per_stratum', type=int, default=6)
    parser.add_argument('--max_cases', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    pack = torch.load(args.failures, map_location='cpu')
    bank = pack['failure_bank']
    out_dir = ensure_dir(args.out)

    if len(bank) == 0:
        summary = {'source': args.failures, 'total_cases': 0, 'scored': 0,
                   'strata': {name: {'count': 0, 'exported': 0} for name, _, _ in STRATA}}
        with open(out_dir / 'scs_strata.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f'{args.failures}: no generated cases; nothing to export.')
        return

    generator = torch.Generator().manual_seed(args.seed)
    if len(bank) > args.max_cases:
        idx = torch.randperm(len(bank), generator=generator)[: args.max_cases].tolist()
        bank = [bank[i] for i in idx]

    scs = SCSComputer(device)
    scored = []
    for item in bank:
        seed_x = item['seed_x']
        if seed_x.ndim == 3:
            seed_x = seed_x.unsqueeze(0)
        gen_x = item['x']
        if gen_x.ndim == 3:
            gen_x = gen_x.unsqueeze(0)
        scored.append({
            'scs': scs.score(seed_x, gen_x),
            'seed_x': seed_x,
            'x': gen_x,
            'seed_idx': int(item['seed_idx']),
            'seed_y': int(item['seed_y']),
            'og': int(item['og']),
            'pred': int(item['pred']),
        })

    summary = {'source': args.failures, 'total_cases': len(pack['failure_bank']),
               'scored': len(scored), 'strata': {}}

    for name, lo, hi in STRATA:
        members = [s for s in scored if lo <= s['scs'] < hi]
        members.sort(key=lambda s: s['scs'], reverse=(name == 'high'))
        take = spread_across_seeds(members, args.per_stratum)
        stratum_dir = ensure_dir(out_dir / name)
        records = []
        for rank, item in enumerate(take):
            stem = f"{rank:02d}_seed{item['seed_idx']}_{item['og']}to{item['pred']}_scs{item['scs']:.3f}"
            save_png(to_image(item['seed_x']), stratum_dir / f'{stem}_original.png')
            save_png(to_image(item['x']), stratum_dir / f'{stem}_generated.png')
            records.append({'file_stem': stem, 'scs': round(item['scs'], 4),
                            'seed_idx': item['seed_idx'], 'seed_label': item['seed_y'],
                            'original_pred': item['og'], 'generated_pred': item['pred']})
        build_pair_grid([(i['seed_x'], i['x']) for i in take], out_dir / f'{name}_pairs.png')
        summary['strata'][name] = {
            'range': [lo if lo > -1 else None, hi if hi < 1.01 else None],
            'count': len(members),
            'proportion': round(len(members) / max(1, len(scored)), 4),
            'exported': len(take),
            'cases': records,
        }

    with open(out_dir / 'scs_strata.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    for name, _, _ in STRATA:
        s = summary['strata'][name]
        print(f"{name:5s} count={s['count']:6d} ({s['proportion']*100:5.1f}%) exported={s['exported']}")
    print(f'-> {out_dir}')


if __name__ == '__main__':
    main()
