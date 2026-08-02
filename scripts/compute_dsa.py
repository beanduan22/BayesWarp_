from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import random
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bayeswarp.utils.config import load_config
from bayeswarp.utils.seed import set_seed
from bayeswarp.utils.io import ensure_dir, save_json
from bayeswarp.utils.device import get_device
from bayeswarp.data.datasets import build_datasets, dataset_meta
from bayeswarp.models.factory import build_model, load_checkpoint
from bayeswarp.models.features import PenultimateExtractor
from bayeswarp.metrics.adequacy import distance_based_surprise_adequacy
from bayeswarp.metrics.statistics import mean_std


def balanced_sample(failure_bank, count: int):
    groups = defaultdict(list)
    for item in failure_bank:
        groups[item['seed_idx']].append(item)
    keys = list(groups.keys())
    selected = []
    while len(selected) < count and keys:
        random.shuffle(keys)
        next_keys = []
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop(0))
            if groups[key]:
                next_keys.append(key)
        keys = next_keys
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--failures', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--num-cases', type=int, default=1000)
    parser.add_argument('--reference-size', type=int, default=10000)
    parser.add_argument('--run', type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    set_seed(cfg['seed'] + args.run)

    train_ds, _ = build_datasets(
        cfg['dataset']['name'],
        cfg['dataset']['root'],
        normalization=cfg['dataset'].get('normalization', 'none'),
        image_size=cfg['dataset'].get('image_size'),
    )
    meta = dataset_meta(cfg['dataset']['name'])
    model = build_model(cfg['model']['name'], meta['num_classes'], pretrained=cfg['model'].get('pretrained', True)).to(device)
    load_checkpoint(model, cfg.get('checkpoint'), device, pretrained=cfg['model'].get('pretrained', True))
    model.eval()

    pack = torch.load(args.failures, map_location='cpu')
    cases = balanced_sample(pack['failure_bank'], args.num_cases)
    if not cases:
        raise RuntimeError('No generated cases available for DSA computation.')

    generator = torch.Generator().manual_seed(cfg['seed'])
    order = torch.randperm(len(train_ds), generator=generator).tolist()[:args.reference_size]
    reference_loader = DataLoader(
        torch.utils.data.Subset(train_ds, order),
        batch_size=cfg['train']['batch_size'],
        shuffle=False,
        num_workers=cfg['train'].get('num_workers', 4),
    )

    extractor = PenultimateExtractor(model)
    reference_features = []
    reference_predictions = []
    with torch.no_grad():
        for xb, _ in tqdm(reference_loader, desc='Reference activations'):
            xb = xb.to(device)
            reference_features.append(extractor(xb).cpu())
            reference_predictions.extend(model(xb).argmax(dim=1).cpu().tolist())
    reference_features = torch.cat(reference_features, dim=0)

    generated_images = [item['x'] for item in cases]
    generated_features = extractor.batched(generated_images, device, batch_size=cfg['train']['batch_size'])
    generated_predictions = [int(item['pred']) for item in cases]
    extractor.close()

    scores = distance_based_surprise_adequacy(
        generated_features,
        generated_predictions,
        reference_features,
        reference_predictions,
    )
    valid = [s for s in scores if s == s]
    mean, std = mean_std(valid)

    out_dir = ensure_dir(cfg['output_dir'])
    save_json(
        {
            'label': args.label,
            'run': args.run,
            'num_cases': len(valid),
            'dsa_mean': mean,
            'dsa_std': std,
            'dsa_scores': valid,
        },
        out_dir / f'dsa_{args.label}_run{args.run}.json',
    )


if __name__ == '__main__':
    main()
