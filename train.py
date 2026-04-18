from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent / 'src'))

import torch
import torch.nn.functional as F
from tqdm import tqdm

from bayeswarp.utils.config import load_config
from bayeswarp.utils.seed import set_seed
from bayeswarp.utils.io import ensure_dir, save_json
from bayeswarp.utils.device import get_device
from bayeswarp.data.datasets import build_loaders, dataset_meta
from bayeswarp.models.factory import build_model


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss_sum += float(loss.item()) * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)
    return {'loss': loss_sum / max(1, total), 'acc': correct / max(1, total)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg['seed'])
    device = get_device()
    train_loader, test_loader = build_loaders(
        cfg['dataset']['name'],
        cfg['dataset']['root'],
        cfg['train']['batch_size'],
        cfg['train'].get('num_workers', 4),
        normalization=cfg['dataset'].get('normalization', 'none'),
        image_size=cfg['dataset'].get('image_size'),
    )
    meta = dataset_meta(cfg['dataset']['name'])
    model = build_model(cfg['model']['name'], meta['num_classes'], pretrained=cfg['model'].get('pretrained', True)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'])

    out_dir = ensure_dir(cfg['output_dir'])
    best_acc = -1.0
    history = []
    for epoch in range(cfg['train']['epochs']):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss.item()))
        metrics = evaluate(model, test_loader, device)
        history.append({'epoch': epoch + 1, **metrics})
        print(metrics)
        if metrics['acc'] > best_acc:
            best_acc = metrics['acc']
            torch.save({'model': model.state_dict(), 'config': cfg}, out_dir / 'best.pt')
    save_json(history, out_dir / 'train_history.json')
    print(f'Saved best checkpoint to {out_dir / "best.pt"}')


if __name__ == '__main__':
    main()
