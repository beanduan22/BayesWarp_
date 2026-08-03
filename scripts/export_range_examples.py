from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import torch

from bayeswarp.utils.config import load_config
from bayeswarp.utils.device import get_device
from bayeswarp.utils.io import ensure_dir
from bayeswarp.data.datasets import pixel_range
from bayeswarp.metrics.quality import SCSComputer, to_unit_range

DATASETS = {
    'mnist': ['results/mnist_lenet4_smoothgrad', 'results/mnist_lenet5_smoothgrad'],
    'cifar10': ['results/cifar10_vgg16_smoothgrad', 'results/cifar10_resnet18_smoothgrad'],
    'imagenet': ['results/imagenet_resnet50_smoothgrad_quick'],
}
METHODS = ('adapt', 'nsgen', 'suntest')
PICKS = ('max', 'median', 'min')


def to_image(x: torch.Tensor, p_min: torch.Tensor, p_max: torch.Tensor) -> torch.Tensor:
    return to_unit_range(x.detach().float(), p_min, p_max).squeeze(0)


def range_for_dir(results_dir: str):
    cfg = load_config(str(Path('configs') / f'{Path(results_dir).name}.yaml'))
    return pixel_range(cfg['dataset']['name'], cfg['dataset'].get('normalization', 'none'))


def as_batch(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0) if x.ndim == 3 else x


def score_bank(path: Path, scs: SCSComputer, sample: int, seed: int, p_min, p_max):
    pack = torch.load(path, map_location='cpu')
    bank = pack['failure_bank']
    if not bank:
        return []
    g = torch.Generator().manual_seed(seed)
    idx = list(range(len(bank)))
    if len(bank) > sample:
        idx = torch.randperm(len(bank), generator=g)[:sample].tolist()
    scored = []
    for i in idx:
        item = bank[i]
        scored.append({
            'scs': scs.score(as_batch(item['seed_x']), as_batch(item['x']), p_min, p_max),
            'bank': str(path),
            'index': i,
        })
    return scored


def load_case(bank_path: str, index: int):
    pack = torch.load(bank_path, map_location='cpu')
    return pack['failure_bank'][index]


def build_groups(by: str):
    groups = {}
    for dataset, dirs in DATASETS.items():
        for d in dirs:
            for m in METHODS:
                p = Path(d) / f'failures_{m}.pt'
                if not p.exists():
                    continue
                key = dataset if by == 'dataset' else f'{Path(d).name}_{m}'
                groups.setdefault(key, []).append(p)
    return groups


def main():
    parser = argparse.ArgumentParser(
        description='Export one original/generated pair per SCS position (max, median, min). '
                    'Group by dataset (pooling every baseline bank) or by method (one group '
                    'per model/method bank).'
    )
    parser.add_argument('--out', default='figures/range_by_dataset')
    parser.add_argument('--by', choices=('dataset', 'method'), default='dataset')
    parser.add_argument('--sample', type=int, default=2000,
                        help='cases sampled per bank before ranking')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    from torchvision.utils import make_grid, save_image

    device = get_device()
    ranges = {d: range_for_dir(d) for dirs in DATASETS.values() for d in dirs}
    default_min, default_max = next(iter(ranges.values()))
    scs = SCSComputer(device, default_min, default_max)
    out_root = ensure_dir(args.out)
    summary = {}

    for group, banks in build_groups(args.by).items():
        scored = []
        for p in banks:
            bank_min, bank_max = ranges[str(p.parent)]
            got = score_bank(p, scs, args.sample, args.seed, bank_min, bank_max)
            print(f'{group}: scored {len(got):5d} from {p}', flush=True)
            scored.extend(got)
        if not scored:
            print(f'{group}: no cases, skipped', flush=True)
            continue

        scored.sort(key=lambda s: s['scs'])
        picks = {
            'max': scored[-1],
            'median': scored[len(scored) // 2],
            'min': scored[0],
        }

        ds_dir = ensure_dir(out_root / group)
        records = []
        rows = []
        for name in PICKS:
            sel = picks[name]
            item = load_case(sel['bank'], sel['index'])
            pick_min, pick_max = ranges[str(Path(sel['bank']).parent)]
            seed_x = to_image(as_batch(item['seed_x']), pick_min, pick_max)
            gen_x = to_image(as_batch(item['x']), pick_min, pick_max)
            if seed_x.size(0) == 1:
                seed_x, gen_x = seed_x.repeat(3, 1, 1), gen_x.repeat(3, 1, 1)
            method = Path(sel['bank']).stem.replace('failures_', '')
            model = Path(sel['bank']).parent.name
            stem = f"{name}_scs{sel['scs']:.3f}_{model}_{method}_{item['og']}to{item['pred']}"
            save_image(seed_x, str(ds_dir / f'{stem}_original.png'))
            save_image(gen_x, str(ds_dir / f'{stem}_generated.png'))
            rows.extend([seed_x, gen_x])
            records.append({
                'position': name,
                'scs': round(sel['scs'], 4),
                'model': model,
                'method': method,
                'seed_label': int(item['seed_y']),
                'original_pred': int(item['og']),
                'generated_pred': int(item['pred']),
                'file_stem': stem,
            })
            print(f"{group:34s} {name:6s} scs={sel['scs']:.4f}  {model}/{method}  "
                  f"{item['og']}->{item['pred']}", flush=True)

        save_image(make_grid(torch.stack(rows), nrow=2, padding=4, pad_value=1.0),
                   str(ds_dir / 'range_pairs.png'))
        summary[group] = {
            'scored': len(scored),
            'scs_min': round(scored[0]['scs'], 4),
            'scs_max': round(scored[-1]['scs'], 4),
            'picks': records,
        }

    with open(out_root / 'range_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print('RANGE EXPORT DONE', flush=True)


if __name__ == '__main__':
    main()
