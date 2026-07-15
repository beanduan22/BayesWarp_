from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent / 'src'))
sys.path.append(str(Path(__file__).resolve().parent))

import torch
from tqdm import tqdm

from bayeswarp.utils.config import load_config
from bayeswarp.utils.seed import set_seed
from bayeswarp.utils.io import ensure_dir, save_json, save_torch
from bayeswarp.utils.device import get_device
from bayeswarp.data.datasets import build_seed_dataset, dataset_meta, select_correctly_classified_seeds
from bayeswarp.models.factory import build_model
from bayeswarp.metrics.failure import compute_failure_metrics
from baselines import BASELINES


def main():
    parser = argparse.ArgumentParser(
        description='Run a baseline testing method under the same controlled '
                    'conditions as BayesWarp (same seeds, oracle, and budget).'
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--baseline', required=True, choices=sorted(BASELINES))
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg['seed'])
    device = get_device()

    seed_ds = build_seed_dataset(
        cfg['dataset']['name'],
        cfg['dataset']['root'],
        normalization=cfg['dataset'].get('normalization', 'none'),
        image_size=cfg['dataset'].get('image_size'),
    )
    meta = dataset_meta(cfg['dataset']['name'])
    model = build_model(cfg['model']['name'], meta['num_classes'], pretrained=cfg['model'].get('pretrained', True)).to(device)
    ckpt = torch.load(cfg['checkpoint'], map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    seed_subset = select_correctly_classified_seeds(
        model, seed_ds, device, num_seeds=cfg['test']['num_seeds'], seed=cfg['seed']
    )

    budget = cfg['bayeswarp']['budget']
    baseline_cfg = dict(cfg.get('baselines', {}).get(args.baseline, {}))
    baseline_cfg.setdefault('eta', cfg['bayeswarp']['eta'])
    baseline = BASELINES[args.baseline](model, device, baseline_cfg)

    baseline.prepare(seed_ds)

    seed_results = []
    failure_bank = []
    for i in tqdm(range(len(seed_subset)), desc=f'{args.baseline}: generating'):
        x, y = seed_subset[i]
        result = baseline.run_on_seed(x, budget)
        result['seed_idx'] = i
        result['seed_y'] = int(y)
        seed_results.append(result)
        for f in result['failures']:
            failure_bank.append({'seed_idx': i, 'seed_x': x.cpu(), 'seed_y': int(y), **f})

    metrics = compute_failure_metrics(seed_results)
    metrics['method'] = args.baseline
    out_dir = ensure_dir(cfg['output_dir'])
    save_torch(
        {'seed_results': seed_results, 'failure_bank': failure_bank, 'metrics': metrics},
        out_dir / f'failures_{args.baseline}.pt',
    )
    save_json(metrics, out_dir / f'metrics_{args.baseline}.json')
    print(metrics)


if __name__ == '__main__':
    main()
