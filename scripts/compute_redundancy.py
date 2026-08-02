from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import torch

from bayeswarp.utils.config import load_config
from bayeswarp.utils.seed import set_seed
from bayeswarp.utils.io import ensure_dir, save_json
from bayeswarp.utils.device import get_device
from bayeswarp.data.datasets import dataset_meta
from bayeswarp.models.factory import build_model, load_checkpoint
from bayeswarp.models.features import PenultimateExtractor
from bayeswarp.metrics.redundancy import pairwise_distance_quantile, redundancy_summary
from bayeswarp.metrics.statistics import median_iqr


def load_vectors(pack, extractor, device, batch_size: int):
    images = [item['x'] for item in pack['failure_bank']]
    if not images:
        raise RuntimeError('No generated cases in the supplied pack.')
    pixel = torch.cat([img if img.ndim == 4 else img.unsqueeze(0) for img in images], dim=0)
    features = extractor.batched(images, device, batch_size=batch_size)
    return pixel, features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--failures', required=True)
    parser.add_argument('--calibration-failures', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--quantile', type=float, default=0.05)
    parser.add_argument('--run', type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    set_seed(cfg['seed'] + args.run)

    meta = dataset_meta(cfg['dataset']['name'])
    model = build_model(cfg['model']['name'], meta['num_classes'], pretrained=cfg['model'].get('pretrained', True)).to(device)
    load_checkpoint(model, cfg.get('checkpoint'), device, pretrained=cfg['model'].get('pretrained', True))
    model.eval()

    extractor = PenultimateExtractor(model)
    batch_size = cfg['train']['batch_size']

    calibration = torch.load(args.calibration_failures, map_location='cpu')
    cal_pixel, cal_feature = load_vectors(calibration, extractor, device, batch_size)
    pixel_threshold = pairwise_distance_quantile(cal_pixel, args.quantile)
    feature_threshold = pairwise_distance_quantile(cal_feature, args.quantile)

    pack = torch.load(args.failures, map_location='cpu')
    pixel, features = load_vectors(pack, extractor, device, batch_size)
    extractor.close()

    summary = redundancy_summary(pixel, features, pixel_threshold, feature_threshold)

    metrics = pack.get('metrics', {})
    cases = [c for c in metrics.get('per_seed_cases', []) if c > 0]
    classes = [d for d, c in zip(metrics.get('per_seed_dof', []), metrics.get('per_seed_cases', [])) if c > 0]
    qff = metrics.get('per_seed_qff', [])

    summary['label'] = args.label
    summary['run'] = args.run
    summary['cases_per_successful_seed'] = median_iqr(cases)
    summary['classes_per_successful_seed'] = median_iqr(classes)
    summary['qff'] = median_iqr(qff)

    out_dir = ensure_dir(cfg['output_dir'])
    save_json(summary, out_dir / f'redundancy_{args.label}_run{args.run}.json')


if __name__ == '__main__':
    main()
