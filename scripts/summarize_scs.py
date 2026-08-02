from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

STRATA = (('high', 0.8, 1.01), ('mid', 0.6, 0.8), ('low', -1.01, 0.6))
QUANTILES = (0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0)


def stats(vals: list[float]) -> dict:
    s = sorted(vals)
    n = len(s)
    q = {f'p{int(p*100)}': round(s[min(int(p * n), n - 1)], 4) for p in QUANTILES}
    q['min'], q['max'] = round(s[0], 4), round(s[-1], 4)
    out = {'n': n, 'quantiles': q, 'strata_counts': {}, 'strata_pct': {}}
    for name, lo, hi in STRATA:
        c = sum(1 for v in s if lo <= v < hi)
        out['strata_counts'][name] = c
        out['strata_pct'][name] = round(100 * c / n, 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='results/_scs_cache')
    ap.add_argument('--out', default='results/_scs_cache/scs_distribution.json')
    args = ap.parse_args()

    by_bank, by_dataset, by_ds_method = {}, defaultdict(list), defaultdict(list)
    for f in sorted(glob.glob(os.path.join(args.cache, '*.jsonl'))):
        rows = [json.loads(l) for l in open(f, encoding='utf-8')]
        name = os.path.basename(f).replace('.jsonl', '')
        vals = [r['scs'] for r in rows]
        by_bank[name] = stats(vals)
        by_bank[name]['distinct_seeds'] = len({r['seed_idx'] for r in rows})
        by_dataset[rows[0]['dataset']].extend(vals)
        by_ds_method[f"{rows[0]['dataset']}__{rows[0]['method']}"].extend(vals)
        print(f'{name}: {len(vals)}')

    report = {
        'note': ('SCS = OpenCLIP ViT-B/32 (openai) cosine between seed and generated image, '
                 'both bilinearly resized to 224. Exhaustive: every case in every bank, no sampling.'),
        'total_cases': sum(len(v) for v in by_dataset.values()),
        'fixed_strata': {n: [lo, hi] for n, lo, hi in STRATA},
        'by_dataset': {k: stats(v) for k, v in by_dataset.items()},
        'by_dataset_method': {k: stats(v) for k, v in by_ds_method.items()},
        'by_bank': by_bank,
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f'\nwrote {args.out}  ({os.path.getsize(args.out)/1024:.1f} KB)')


if __name__ == '__main__':
    main()
