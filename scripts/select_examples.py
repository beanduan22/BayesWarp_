from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import torch

from bayeswarp.utils.io import ensure_dir

STRATA_FIXED = (('high', 0.8, 1.01), ('mid', 0.6, 0.8), ('low', -1.01, 0.6))
DATASETS = ('mnist', 'cifar10', 'imagenet')
PER_STRATUM = 2


def load_scores(cache: Path) -> list[dict]:
    rows = []
    for f in sorted(cache.glob('*.jsonl')):
        with open(f, encoding='utf-8') as fh:
            rows.extend(json.loads(line) for line in fh)
    return rows


def strata_for(rows: list[dict], fallback: str) -> tuple[tuple[str, float, float], ...]:
    counts = {name: sum(1 for r in rows if lo <= r['scs'] < hi) for name, lo, hi in STRATA_FIXED}
    if all(c > 0 for c in counts.values()) or fallback != 'tertile':
        return STRATA_FIXED
    vals = sorted(r['scs'] for r in rows)
    q1 = vals[len(vals) // 3]
    q2 = vals[2 * len(vals) // 3]
    return (('high', q2, 1.01), ('mid', q1, q2), ('low', -1.01, q1))


def pick_two(members: list[dict]) -> list[dict]:
    med = median(r['scs'] for r in members)
    by_seed: dict[int, list[dict]] = {}
    for r in members:
        by_seed.setdefault(r['seed_idx'], []).append(r)

    reps = []
    for seed, group in by_seed.items():
        rep = min(group, key=lambda r: (abs(r['scs'] - med), r['case_id']))
        reps.append(rep)
    reps.sort(key=lambda r: (abs(r['scs'] - med), r['case_id']))
    return reps[:PER_STRATUM], med, len(by_seed)


def to_image(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        x = x.squeeze(0)
    x = x.detach().float()
    lo, hi = float(x.min()), float(x.max())
    if hi - lo > 1e-8:
        x = (x - lo) / (hi - lo)
    return x.clamp(0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='results/_scs_cache')
    ap.add_argument('--out', default='figures/selected')
    ap.add_argument('--fallback', choices=('tertile', 'none'), default='tertile')
    args = ap.parse_args()

    from torchvision.utils import save_image

    rows = load_scores(Path(args.cache))
    print(f'loaded {len(rows)} scored cases')

    manifest = {}
    wanted: dict[str, list[dict]] = {}

    for dataset in DATASETS:
        ds_rows = [r for r in rows if r['dataset'] == dataset]
        if not ds_rows:
            continue
        strata = strata_for(ds_rows, args.fallback)
        vals = sorted(r['scs'] for r in ds_rows)
        entry = {
            'scored': len(ds_rows),
            'scs_min': round(vals[0], 4),
            'scs_max': round(vals[-1], 4),
            'thresholds': {n: [round(lo, 4), round(hi, 4)] for n, lo, hi in strata},
            'thresholds_source': 'fixed' if strata is STRATA_FIXED else 'tertile (a fixed stratum was empty)',
            'strata': {},
        }

        for name, lo, hi in strata:
            members = [r for r in ds_rows if lo <= r['scs'] < hi]
            if not members:
                entry['strata'][name] = {'count': 0, 'distinct_seeds': 0, 'picks': [],
                                         'note': 'empty — nothing to select'}
                print(f'{dataset:9s} {name:5s} count=0  (empty)')
                continue
            picks, med, n_seeds = pick_two(members)
            if len(picks) < PER_STRATUM:
                note = f'only {n_seeds} distinct seed(s); fewer than {PER_STRATUM} picks possible'
            else:
                note = None
            entry['strata'][name] = {
                'count': len(members),
                'distinct_seeds': n_seeds,
                'stratum_median_scs': round(med, 4),
                'picks': [{
                    'case_id': p['case_id'],
                    'seed_idx': p['seed_idx'],
                    'seed_label': p['seed_y'],
                    'scs': round(p['scs'], 4),
                    'class_change': f"{p['og']}->{p['pred']}",
                    'original_pred': p['og'],
                    'generated_pred': p['pred'],
                    'model': p['model'],
                    'method': p['method'],
                } for p in picks],
            }
            if note:
                entry['strata'][name]['note'] = note
            wanted.setdefault(dataset, []).extend((name, p) for p in picks)
            print(f'{dataset:9s} {name:5s} count={len(members):7d} seeds={n_seeds:4d} '
                  f'median={med:.4f} -> ' +
                  ', '.join(f"{p['case_id']}(scs={p['scs']:.4f},seed={p['seed_idx']})" for p in picks))
        manifest[dataset] = entry

    out_root = ensure_dir(args.out)
    need_banks: dict[str, list[tuple[str, str, dict]]] = {}
    for dataset, items in wanted.items():
        for stratum, p in items:
            bank = f"results/{p['model']}/failures_{p['method']}.pt"
            need_banks.setdefault(bank, []).append((dataset, stratum, p))

    for bank, items in need_banks.items():
        print(f'loading {bank} for {len(items)} pick(s)')
        pack = torch.load(bank, map_location='cpu')
        for dataset, stratum, p in items:
            it = pack['failure_bank'][p['index']]
            d = ensure_dir(out_root / dataset / stratum)
            stem = (f"{dataset}_{stratum}_scs{p['scs']:.3f}_seed{p['seed_idx']}"
                    f"_{p['og']}to{p['pred']}_{p['model']}_{p['method']}")
            seed_x, gen_x = to_image(it['seed_x']), to_image(it['x'])
            if seed_x.ndim == 3 and seed_x.size(0) == 1:
                seed_x, gen_x = seed_x.repeat(3, 1, 1), gen_x.repeat(3, 1, 1)
            save_image(seed_x, str(d / f'{stem}_original.png'))
            save_image(gen_x, str(d / f'{stem}_generated.png'))
        del pack

    with open(out_root / 'selection.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    total = sum(len(v) for v in wanted.values())
    print(f'\nSELECTION DONE — {total} cases -> {out_root}')


if __name__ == '__main__':
    main()
