from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent / 'src'))

import torch

from bayeswarp.utils.config import load_config
from bayeswarp.utils.seed import set_seed
from bayeswarp.utils.io import save_json
from bayeswarp.utils.device import get_device
from bayeswarp.data.datasets import dataset_meta
from bayeswarp.models.factory import build_model
from bayeswarp.metrics.quality import compute_fid, compute_scs
from bayeswarp.metrics.coverage import neuron_coverage, topk_neuron_coverage, critical_neuron_coverage


def default_topk(dataset_name: str) -> int:
    name = dataset_name.lower()
    if name == 'mnist':
        return 3
    if name == 'cifar10':
        return 5
    if name == 'imagenet':
        return 30
    return 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--failures', required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg['seed'])
    device = get_device()
    meta = dataset_meta(cfg['dataset']['name'])
    model = build_model(cfg['model']['name'], meta['num_classes'], pretrained=cfg['model'].get('pretrained', True)).to(device)
    ckpt = torch.load(cfg['checkpoint'], map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    pack = torch.load(args.failures, map_location='cpu')
    failure_bank = pack['failure_bank']
    if len(failure_bank) == 0:
        raise RuntimeError('No failures to evaluate.')

    real_images = [f['seed_x'].unsqueeze(0) if f['seed_x'].ndim == 3 else f['seed_x'] for f in failure_bank]
    fake_images = [f['x'] for f in failure_bank]
    pairs = [((f['seed_x'].unsqueeze(0) if f['seed_x'].ndim == 3 else f['seed_x']), f['x']) for f in failure_bank]

    metrics = dict(pack.get('metrics', {}))
    try:
        metrics['FID'] = compute_fid(real_images, fake_images, device)
    except Exception as e:
        metrics['FID'] = f'Unavailable: {e}'
    try:
        metrics['SCS'] = compute_scs(pairs, device)
    except Exception as e:
        metrics['SCS'] = f'Unavailable: {e}'

    cov_images = [x.to(device) for x in fake_images[: min(256, len(fake_images))]]
    metrics['NC'] = neuron_coverage(model, cov_images)
    metrics['TKNC'] = topk_neuron_coverage(model, cov_images, k=default_topk(cfg['dataset']['name']))
    metrics['CNC'] = critical_neuron_coverage(model, cov_images)

    out_path = Path(cfg['output_dir']) / 'evaluation_metrics.json'
    save_json(metrics, out_path)
    print(metrics)


if __name__ == '__main__':
    main()
