from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent / 'src'))

import torch
from tqdm import tqdm

from bayeswarp.utils.config import load_config
from bayeswarp.utils.seed import set_seed
from bayeswarp.utils.io import ensure_dir, save_json, save_torch
from bayeswarp.utils.device import get_device
from bayeswarp.data.datasets import build_datasets, dataset_meta, select_correctly_classified_seeds
from bayeswarp.models.factory import build_model
from bayeswarp.testing.bayeswarp import BayesWarpTester, BayesWarpConfig
from bayeswarp.metrics.failure import compute_failure_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ablation', default='none', choices=['none', 'no_localization', 'no_bayesian'])
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg['seed'])
    device = get_device()

    train_ds, _ = build_datasets(
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

    seed_subset = select_correctly_classified_seeds(model, train_ds, device, num_seeds=cfg['test']['num_seeds'])
    bw_cfg = BayesWarpConfig(ablation=args.ablation, **cfg['bayeswarp'])
    tester = BayesWarpTester(model, device, bw_cfg)

    seed_results = []
    failure_bank = []
    for i in tqdm(range(len(seed_subset)), desc='Generating failures'):
        x, y = seed_subset[i]
        result = tester.run_on_seed(x)
        result['seed_idx'] = i
        result['seed_y'] = int(y)
        seed_results.append(result)
        for f in result['failures']:
            failure_bank.append({
                'seed_idx': i,
                'seed_x': x.cpu(),
                'seed_y': int(y),
                **f,
            })

    metrics = compute_failure_metrics(seed_results)
    out_dir = ensure_dir(cfg['output_dir'])
    suffix = args.ablation if args.ablation != 'none' else 'main'
    save_torch({'seed_results': seed_results, 'failure_bank': failure_bank, 'metrics': metrics}, out_dir / f'failures_{suffix}.pt')
    save_json(metrics, out_dir / f'metrics_{suffix}.json')
    print(metrics)


if __name__ == '__main__':
    main()
