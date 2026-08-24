from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import torch

from bayeswarp.utils.config import load_config
from bayeswarp.utils.device import get_device
from bayeswarp.utils.io import ensure_dir, save_json
from bayeswarp.utils.seed import set_seed
from bayeswarp.data.datasets import build_seed_dataset, pixel_range
from bayeswarp.finetune.sources import bank_seed_ids, build_manifest
from bayeswarp.pipeline import build_seeds, build_subject


def pack_cache_key(path: Path) -> str:
    resolved = path.resolve()
    return hashlib.sha256(f'{resolved}|{resolved.stat().st_size}'.encode('utf-8')).hexdigest()[:16]


def load_scs_scores(path: Path, expected: int):
    payload = path.read_text(encoding='utf-8')
    if path.suffix == '.jsonl':
        scores = [float(json.loads(line)['scs']) for line in payload.splitlines() if line.strip()]
    else:
        data = json.loads(payload)
        scores = [float(v) for v in (data['scores'] if isinstance(data, dict) else data)]
    if len(scores) != expected:
        raise ValueError(f'{path} has {len(scores)} SCS scores but the pack has {expected} cases')
    return scores


def compute_scs_scores(pack, cfg, device, cache_dir: Path, cache_key: str):
    from tqdm import tqdm
    from bayeswarp.metrics.quality import SCSComputer

    cache_path = cache_dir / f'finetune_sources__{cache_key}.json'
    bank = pack['failure_bank']
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding='utf-8'))
        if len(cached['scores']) == len(bank):
            return [float(v) for v in cached['scores']], str(cache_path)

    p_min, p_max = pixel_range(cfg['dataset']['name'], cfg['dataset'].get('normalization', 'none'))
    scs = SCSComputer(device, p_min, p_max)
    scores = [
        float(scs.score(
            item['seed_x'].unsqueeze(0) if item['seed_x'].ndim == 3 else item['seed_x'],
            item['x'],
        ))
        for item in tqdm(bank, desc='Scoring SCS')
    ]
    save_json({'cases': len(bank), 'scores': scores}, cache_path)
    return scores, str(cache_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument(
        '--failures', nargs='+', required=True,
        help='Every failure pack the generated conditions will draw from, across all runs. '
             'A source is eligible only if it supplies enough cases in all of them.',
    )
    parser.add_argument('--out', required=True)
    parser.add_argument('--num-sources', type=int, default=100)
    parser.add_argument('--per-source', type=int, default=10)
    parser.add_argument('--scs-threshold', type=float, default=0.8)
    parser.add_argument(
        '--scs-scores', nargs='*', default=None,
        help='Precomputed SCS scores, one file per --failures pack (.json list or .jsonl rows).',
    )
    parser.add_argument('--scs-cache', default='results/_scs_cache')
    parser.add_argument(
        '--skip-scs', action='store_true',
        help='Do not enforce high-SCS availability. Use only when the high_scs condition is not run.',
    )
    parser.add_argument('--selection-seed', type=int, default=None)
    args = parser.parse_args()

    if args.scs_scores and len(args.scs_scores) != len(args.failures):
        raise SystemExit('--scs-scores must have one entry per --failures pack')

    cfg = load_config(args.config)
    selection_seed = int(cfg['seed'] if args.selection_seed is None else args.selection_seed)
    device = get_device()
    set_seed(cfg['seed'])

    seed_split = cfg['dataset'].get('seed_split', 'train')
    seed_dataset = build_seed_dataset(
        cfg['dataset']['name'],
        cfg['dataset']['root'],
        normalization=cfg['dataset'].get('normalization', 'none'),
        image_size=cfg['dataset'].get('image_size'),
        split=seed_split,
    )

    model = build_subject(cfg, device)
    fallback_indices = [int(i) for i in build_seeds(cfg, model, device).indices]
    del model

    cache_dir = ensure_dir(args.scs_cache)
    per_source = int(args.per_source)

    eligible = None
    pack_reports = []
    for position, failures_path in enumerate(args.failures):
        path = Path(failures_path)
        pack = torch.load(path, map_location='cpu')
        bank = pack['failure_bank']
        if not bank:
            raise SystemExit(f'{path} contains no generated cases')
        ids = bank_seed_ids(pack, fallback_indices)

        total_counts = {}
        for seed_id in ids:
            total_counts[seed_id] = total_counts.get(seed_id, 0) + 1

        scs_source = None
        if args.skip_scs:
            high_counts = total_counts
        else:
            if args.scs_scores:
                scores = load_scs_scores(Path(args.scs_scores[position]), len(bank))
                scs_source = args.scs_scores[position]
            else:
                scores, scs_source = compute_scs_scores(
                    pack, cfg, device, Path(cache_dir), pack_cache_key(path)
                )
            high_counts = {}
            for seed_id, score in zip(ids, scores):
                if float(score) >= float(args.scs_threshold):
                    high_counts[seed_id] = high_counts.get(seed_id, 0) + 1

        qualifying = {
            seed_id
            for seed_id, count in total_counts.items()
            if count >= per_source and high_counts.get(seed_id, 0) >= per_source
        }
        eligible = qualifying if eligible is None else (eligible & qualifying)
        pack_reports.append({
            'pack': str(path),
            'cases': len(bank),
            'seeds_with_cases': len(total_counts),
            f'seeds_with_at_least_{per_source}_cases': sum(
                1 for c in total_counts.values() if c >= per_source
            ),
            f'seeds_with_at_least_{per_source}_high_scs_cases': sum(
                1 for c in high_counts.values() if c >= per_source
            ),
            'scs_source': scs_source,
            'running_eligible': len(eligible),
        })
        del pack

    eligible = sorted(eligible or set())
    if len(eligible) < int(args.num_sources):
        detail = '\n'.join(json.dumps(r) for r in pack_reports)
        raise SystemExit(
            f'Only {len(eligible)} sources supply >= {per_source} cases (and >= {per_source} '
            f'cases above SCS {args.scs_threshold}) in every pack, but --num-sources is '
            f'{args.num_sources}.\nLower --num-sources or --per-source, or widen the search '
            f'budget.\nPer-pack availability:\n{detail}'
        )

    rng = random.Random(selection_seed)
    chosen = sorted(rng.sample(eligible, int(args.num_sources)))
    sources = [(index, int(seed_dataset[index][1])) for index in chosen]

    manifest = build_manifest(
        dataset=cfg['dataset']['name'],
        seed_split=seed_split,
        model=cfg['model']['name'],
        config_path=args.config,
        sources=sources,
        per_source=per_source,
        selection_seed=selection_seed,
        eligibility={
            'scs_threshold': None if args.skip_scs else float(args.scs_threshold),
            'eligible_sources': len(eligible),
            'packs': pack_reports,
        },
    )
    save_json(manifest, args.out)
    print(
        f'Wrote {args.out}: {manifest["num_sources"]} sources x {manifest["per_source"]} '
        f'= {manifest["num_samples"]} added samples per condition; '
        f'fingerprint {manifest["fingerprint"]}'
    )


if __name__ == '__main__':
    main()
