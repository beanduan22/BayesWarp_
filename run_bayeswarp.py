from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent / 'src'))

from bayeswarp.pipeline import generate
from bayeswarp.testing.bayeswarp import ABLATIONS
from bayeswarp.utils.config import load_config
from bayeswarp.utils.device import get_device
from bayeswarp.utils.io import ensure_dir, save_json, save_torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ablation', default='none', choices=list(ABLATIONS))
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--num-seeds', type=int, default=None)
    parser.add_argument('--suffix', default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()

    seed_results, failure_bank, metrics = generate(
        cfg,
        device,
        ablation=args.ablation,
        run=args.run,
        skip=args.skip,
        num_seeds=args.num_seeds,
    )

    base = args.suffix or (args.ablation if args.ablation != 'none' else 'main')
    out_dir = ensure_dir(cfg['output_dir'])
    save_torch(
        {'seed_results': seed_results, 'failure_bank': failure_bank, 'metrics': metrics},
        out_dir / f'failures_{base}_run{args.run}.pt',
    )
    save_json(metrics, out_dir / f'metrics_{base}_run{args.run}.json')


if __name__ == '__main__':
    main()
