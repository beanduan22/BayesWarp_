from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import random
import sys
sys.path.append(str(Path(__file__).resolve().parent / 'src'))

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm


class FailureDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y.long()

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, idx: int):
        return self.X[idx], int(self.y[idx].item())

from bayeswarp.utils.config import load_config
from bayeswarp.utils.seed import set_seed
from bayeswarp.utils.io import ensure_dir, save_json
from bayeswarp.utils.device import get_device
from bayeswarp.data.datasets import build_datasets, build_loaders, dataset_meta
from bayeswarp.models.factory import build_model


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


def balanced_failure_subset(failure_bank, num_failures: int):
    groups = defaultdict(list)
    for item in failure_bank:
        groups[(item['seed_idx'], item['target_class'])].append(item)
    keys = list(groups.keys())
    selected = []
    while len(selected) < num_failures and keys:
        random.shuffle(keys)
        next_keys = []
        for k in keys:
            if groups[k] and len(selected) < num_failures:
                selected.append(groups[k].pop(0))
            if groups[k]:
                next_keys.append(k)
        keys = next_keys
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--failures', required=True)
    parser.add_argument('--num_failures', type=int, default=1000)
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
    ckpt = torch.load(cfg['checkpoint'], map_location=device)
    model.load_state_dict(ckpt['model'])

    before_acc = evaluate(model, test_loader, device)
    pack = torch.load(args.failures, map_location='cpu')
    failure_bank = pack['failure_bank']
    if len(failure_bank) == 0:
        raise RuntimeError('No failures found in the supplied file.')

    if len(failure_bank) > args.num_failures:
        failure_bank = balanced_failure_subset(failure_bank, args.num_failures)

    X = torch.cat([f['x'] for f in failure_bank], dim=0)
    y = torch.tensor([f['seed_y'] for f in failure_bank], dtype=torch.long)
    failure_ds = FailureDataset(X, y)
    mixed_ds = ConcatDataset([train_ds, failure_ds])
    mixed_loader = DataLoader(mixed_ds, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=cfg['train'].get('num_workers', 4))

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'])
    epochs = cfg['finetune'].get('epochs', 10)
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(mixed_loader, desc=f'Finetune {epoch+1}/{epochs}')
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss.item()))

    after_acc = evaluate(model, test_loader, device)
    out_dir = ensure_dir(cfg['output_dir'])
    torch.save({'model': model.state_dict(), 'config': cfg}, out_dir / 'finetuned.pt')
    result = {'acc_before': before_acc, 'acc_after': after_acc, 'delta_acc': after_acc - before_acc}
    save_json(result, out_dir / 'finetune_metrics.json')
    print(result)


if __name__ == '__main__':
    main()
