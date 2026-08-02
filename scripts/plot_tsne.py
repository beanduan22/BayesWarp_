from __future__ import annotations
import argparse
from pathlib import Path
import random
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

from bayeswarp.utils.config import load_config
from bayeswarp.utils.seed import set_seed
from bayeswarp.utils.io import ensure_dir
from bayeswarp.utils.device import get_device
from bayeswarp.data.datasets import build_datasets, dataset_meta
from bayeswarp.models.factory import build_model, load_checkpoint
from bayeswarp.models.features import PenultimateExtractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--failures', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--num-cases', type=int, default=1000)
    parser.add_argument('--num-reference', type=int, default=2000)
    parser.add_argument('--perplexity', type=float, default=30.0)
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

    generator = torch.Generator().manual_seed(cfg['seed'])
    order = torch.randperm(len(train_ds), generator=generator).tolist()[:args.num_reference]
    loader = DataLoader(Subset(train_ds, order), batch_size=cfg['train']['batch_size'], shuffle=False)

    extractor = PenultimateExtractor(model)
    reference = []
    for xb, _ in loader:
        reference.append(extractor(xb.to(device)).cpu())
    reference = torch.cat(reference, dim=0)

    pack = torch.load(args.failures, map_location='cpu')
    bank = pack['failure_bank']
    rng = random.Random(cfg['seed'] + args.run)
    sample = bank if len(bank) <= args.num_cases else rng.sample(bank, args.num_cases)
    generated = extractor.batched([item['x'] for item in sample], device, batch_size=cfg['train']['batch_size'])
    extractor.close()

    features = torch.cat([reference, generated], dim=0).numpy()
    embedding = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        init='pca',
        random_state=cfg['seed'] + args.run,
    ).fit_transform(features)

    split = reference.size(0)
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    ax.scatter(embedding[:split, 0], embedding[:split, 1], s=4, alpha=0.35, label='Original')
    ax.scatter(embedding[split:, 0], embedding[split:, 1], s=4, alpha=0.55, label='Generated')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='best', frameon=False, fontsize=8)
    fig.tight_layout()

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == '__main__':
    main()
