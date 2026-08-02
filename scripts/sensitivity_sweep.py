from __future__ import annotations
import argparse
import copy
from pathlib import Path
import random
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import torch

from bayeswarp.pipeline import generate
from bayeswarp.utils.config import load_config
from bayeswarp.utils.device import get_device
from bayeswarp.utils.io import ensure_dir, save_json
from bayeswarp.metrics.quality import compute_fid, compute_scs


SWEEPS = {
    'alpha': [0.05, 0.1, 0.2],
    'rho': [0.4, 0.6, 0.8],
    'S': [16, 32, 64],
    'm': [32, 64, 128],
    'n': [1, 2, 4],
}


def subsample(failure_bank, count: int, seed: int):
    if len(failure_bank) <= count:
        return list(failure_bank)
    rng = random.Random(seed)
    return rng.sample(list(failure_bank), count)


def quality(failure_bank, device, count: int, seed: int):
    sample = subsample(failure_bank, count, seed)
    real = [item['seed_x'].unsqueeze(0) if item['seed_x'].ndim == 3 else item['seed_x'] for item in sample]
    fake = [item['x'] for item in sample]
    pairs = list(zip(real, fake))
    try:
        fid = compute_fid(real, fake, device)
    except Exception as e:
        fid = f'Unavailable: {e}'
    try:
        scs = compute_scs(pairs, device)
    except Exception as e:
        scs = f'Unavailable: {e}'
    return fid, scs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--parameter', required=True, choices=sorted(SWEEPS))
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--num-seeds', type=int, default=None)
    parser.add_argument('--quality-samples', type=int, default=1000)
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    device = get_device()
    out_dir = ensure_dir(base_cfg['output_dir'])

    results = []
    for value in SWEEPS[args.parameter]:
        cfg = copy.deepcopy(base_cfg)
        cfg['bayeswarp'][args.parameter] = value
        _, failure_bank, metrics = generate(
            cfg,
            device,
            run=args.run,
            num_seeds=args.num_seeds,
            progress=f'{args.parameter}={value}',
        )
        fid, scs = quality(failure_bank, device, args.quality_samples, cfg['seed'] + args.run)
        results.append({
            'parameter': args.parameter,
            'value': value,
            'NoF': metrics['NoF'],
            'FSR': metrics['FSR'],
            'DoF': metrics['DoF'],
            'FID': fid,
            'SCS': scs,
        })

    save_json(results, out_dir / f'sensitivity_{args.parameter}_run{args.run}.json')


if __name__ == '__main__':
    main()
