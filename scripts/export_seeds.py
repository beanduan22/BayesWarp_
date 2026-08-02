from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import torch

from bayeswarp.utils.config import load_config
from bayeswarp.utils.seed import set_seed
from bayeswarp.utils.io import save_json
from bayeswarp.utils.device import get_device
from bayeswarp.pipeline import build_seeds, build_subject


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--num-seeds', type=int, default=None)
    parser.add_argument('--runs', type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    set_seed(cfg['seed'])

    model = build_subject(cfg, device)
    subset = build_seeds(cfg, model, device, skip=args.skip, num_seeds=args.num_seeds)

    entries = []
    for position, dataset_index in enumerate(subset.indices):
        _, label = subset.dataset[dataset_index]
        entries.append({
            'position': position,
            'dataset_index': int(dataset_index),
            'label': int(label),
        })

    save_json(
        {
            'config': args.config,
            'dataset': cfg['dataset']['name'],
            'split': cfg['dataset'].get('seed_split', 'train'),
            'model': cfg['model']['name'],
            'selection_seed': int(cfg['seed']),
            'search_seeds': [int(cfg['seed']) + run for run in range(args.runs)],
            'skip': int(args.skip),
            'num_seeds': len(entries),
            'seeds': entries,
        },
        args.out,
    )


if __name__ == '__main__':
    main()
