from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from bayeswarp.utils.config import load_config
from bayeswarp.utils.seed import set_seed
from bayeswarp.utils.io import ensure_dir, save_json
from bayeswarp.utils.device import get_device
from bayeswarp.data.datasets import build_datasets, build_loaders, dataset_meta, pixel_range
from bayeswarp.models.factory import build_model, load_checkpoint
from bayeswarp.metrics.quality import SCSComputer
from bayeswarp.finetune.conditions import (
    CONDITIONS,
    autoattack_condition,
    continued_condition,
    random_localized_condition,
    select_generated,
    source_samples,
    standard_condition,
)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return correct / max(1, total)


def build_added_samples(args, cfg, model, device, train_ds):
    dataset_name = cfg['dataset']['name']
    normalization = cfg['dataset'].get('normalization', 'none')
    bw = cfg['bayeswarp']

    if args.condition in ('high_scs', 'bayeswarp'):
        pack = torch.load(args.failures, map_location='cpu')
        failure_bank = pack['failure_bank']
        if args.condition == 'bayeswarp':
            return select_generated(failure_bank, args.num_samples)
        p_min, p_max = pixel_range(dataset_name, normalization)
        scs = SCSComputer(device, p_min, p_max)
        scores = [
            scs.score(
                item['seed_x'].unsqueeze(0) if item['seed_x'].ndim == 3 else item['seed_x'],
                item['x'],
            )
            for item in tqdm(failure_bank, desc='Scoring SCS')
        ]
        return select_generated(failure_bank, args.num_samples, scs_scores=scores, scs_threshold=args.scs_threshold)

    X, y = source_samples(train_ds, args.num_samples, cfg['seed'])

    if args.condition == 'continued':
        return continued_condition(X, y)
    if args.condition == 'standard':
        return standard_condition(X, y, dataset_name)
    if args.condition == 'random':
        return random_localized_condition(
            model,
            device,
            X,
            y,
            dataset_name=dataset_name,
            normalization=normalization,
            saliency_method=bw['saliency_method'],
            alpha=bw['alpha'],
            area_min=bw['area_min'],
            tau_iou=bw['tau_iou'],
            d_max=bw['d_max'],
            rho=bw['rho'],
            n=bw['n'],
            r=bw['r'],
            eta=bw['eta'],
        )
    if args.condition == 'autoattack':
        return autoattack_condition(
            model,
            device,
            X,
            y,
            dataset_name=dataset_name,
            normalization=normalization,
            epsilon=args.epsilon,
        )
    raise ValueError(f'Unsupported condition: {args.condition}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--condition', required=True, choices=list(CONDITIONS))
    parser.add_argument('--failures', default=None)
    parser.add_argument('--num-samples', type=int, default=1000)
    parser.add_argument('--scs-threshold', type=float, default=0.8)
    parser.add_argument('--epsilon', type=float, default=8.0 / 255.0)
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
    _, test_loader = build_loaders(
        cfg['dataset']['name'],
        cfg['dataset']['root'],
        cfg['train']['batch_size'],
        cfg['train'].get('num_workers', 4),
        normalization=cfg['dataset'].get('normalization', 'none'),
        image_size=cfg['dataset'].get('image_size'),
    )

    meta = dataset_meta(cfg['dataset']['name'])
    model = build_model(cfg['model']['name'], meta['num_classes'], pretrained=cfg['model'].get('pretrained', True)).to(device)
    load_checkpoint(model, cfg.get('checkpoint'), device, pretrained=cfg['model'].get('pretrained', True))

    before_acc = evaluate(model, test_loader, device)

    X, y = build_added_samples(args, cfg, model, device, train_ds)
    added_ds = TensorDataset(X, y)
    mixed_ds = ConcatDataset([train_ds, added_ds])
    mixed_loader = DataLoader(
        mixed_ds,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['train'].get('num_workers', 4),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'])
    epochs = cfg['finetune'].get('epochs', 10)
    for epoch in range(epochs):
        model.train()
        for xb, yb in tqdm(mixed_loader, desc=f'{args.condition} {epoch + 1}/{epochs}'):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()

    after_acc = evaluate(model, test_loader, device)

    out_dir = ensure_dir(cfg['output_dir'])
    save_json(
        {
            'condition': args.condition,
            'run': args.run,
            'num_added': int(X.size(0)),
            'acc_before': before_acc,
            'acc_after': after_acc,
            'delta_acc': after_acc - before_acc,
        },
        out_dir / f'controlled_finetune_{args.condition}_run{args.run}.json',
    )


if __name__ == '__main__':
    main()
