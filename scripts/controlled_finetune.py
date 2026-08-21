"""Controlled fine-tuning study: one condition per invocation, shared sources.

Every condition adds ``num_sources * per_source`` samples derived from the same
manifest of source samples, so identity, count and class distribution are held
fixed across conditions by construction. The manifest fingerprint is written to
the result file; ``scripts/verify_finetune_protocol.py`` checks that all six
conditions in a directory really did use the same one.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from tqdm import tqdm

from bayeswarp.utils.config import load_config
from bayeswarp.utils.seed import set_seed
from bayeswarp.utils.io import ensure_dir, save_json
from bayeswarp.utils.device import get_device
from bayeswarp.data.datasets import (
    build_datasets,
    build_loaders,
    build_seed_dataset,
    dataset_meta,
    pixel_range,
)
from bayeswarp.models.factory import build_model, load_checkpoint
from bayeswarp.metrics.quality import SCSComputer
from bayeswarp.finetune.conditions import (
    CONDITIONS,
    GENERATED_CONDITIONS,
    autoattack_condition,
    continued_condition,
    group_bank_by_source,
    random_localized_condition,
    select_generated,
    standard_condition,
)
from bayeswarp.finetune.sources import (
    bank_seed_ids,
    class_histogram,
    load_manifest,
    source_indices,
    source_tensors,
)
from bayeswarp.pipeline import build_seeds


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


def load_scs_scores(path: Path, expected: int):
    payload = path.read_text(encoding='utf-8')
    if path.suffix == '.jsonl':
        scores = [float(json.loads(line)['scs']) for line in payload.splitlines() if line.strip()]
    else:
        data = json.loads(payload)
        scores = [float(v) for v in (data['scores'] if isinstance(data, dict) else data)]
    if len(scores) != expected:
        raise ValueError(f'{path} has {len(scores)} SCS scores but the pack has {expected} cases')
    return scores


def build_added_samples(args, cfg, manifest, model, device, X_src, y_src):
    dataset_name = cfg['dataset']['name']
    normalization = cfg['dataset'].get('normalization', 'none')
    bw = cfg['bayeswarp']
    per_source = int(manifest['per_source'])
    derivation_seed = int(cfg['seed']) + int(args.run)

    if args.condition in GENERATED_CONDITIONS:
        if not args.failures:
            raise SystemExit(f'--failures is required for the {args.condition} condition')
        pack = torch.load(args.failures, map_location='cpu')
        failure_bank = pack['failure_bank']
        fallback = [int(i) for i in build_seeds(cfg, model, device).indices]
        seed_ids = bank_seed_ids(pack, fallback)

        scores = None
        threshold = None
        if args.condition == 'high_scs':
            threshold = float(args.scs_threshold)
            if args.scs_scores:
                scores = load_scs_scores(Path(args.scs_scores), len(failure_bank))
            else:
                p_min, p_max = pixel_range(dataset_name, normalization)
                scs = SCSComputer(device, p_min, p_max)
                scores = [
                    float(scs.score(
                        item['seed_x'].unsqueeze(0) if item['seed_x'].ndim == 3 else item['seed_x'],
                        item['x'],
                    ))
                    for item in tqdm(failure_bank, desc='Scoring SCS')
                ]

        groups = group_bank_by_source(
            failure_bank, seed_ids, scs_scores=scores, scs_threshold=threshold
        )
        X, y, info = select_generated(
            groups, source_indices(manifest), per_source, seed=derivation_seed
        )
        info['failures'] = args.failures
        if threshold is not None:
            info['scs_threshold'] = threshold
        return X, y, info

    if args.condition == 'continued':
        return continued_condition(X_src, y_src, per_source)
    if args.condition == 'standard':
        return standard_condition(X_src, y_src, dataset_name, per_source, seed=derivation_seed)
    if args.condition == 'random':
        return random_localized_condition(
            model,
            device,
            X_src,
            y_src,
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
            per_source=per_source,
            seed=derivation_seed,
        )
    if args.condition == 'autoattack':
        return autoattack_condition(
            model,
            device,
            X_src,
            y_src,
            dataset_name=dataset_name,
            normalization=normalization,
            per_source=per_source,
            epsilon=args.epsilon,
            seed=derivation_seed,
            num_classes=dataset_meta(dataset_name)['num_classes'],
        )
    raise ValueError(f'Unsupported condition: {args.condition}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--condition', required=True, choices=list(CONDITIONS))
    parser.add_argument(
        '--sources', required=True,
        help='Source manifest from scripts/build_finetune_sources.py, shared by all conditions.',
    )
    parser.add_argument('--failures', default=None)
    parser.add_argument('--scs-scores', default=None)
    parser.add_argument('--scs-threshold', type=float, default=0.8)
    parser.add_argument('--epsilon', type=float, default=8.0 / 255.0)
    parser.add_argument('--run', type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest = load_manifest(args.sources)
    if manifest['dataset'] != cfg['dataset']['name'] or manifest['model'] != cfg['model']['name']:
        raise SystemExit(
            f'Source manifest was built for {manifest["model"]}/{manifest["dataset"]}, '
            f'but this config is {cfg["model"]["name"]}/{cfg["dataset"]["name"]}'
        )

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

    # Sources come from the deterministic evaluation transform, exactly as the
    # generation seeds do. Reading them from the training dataset would apply a
    # random crop/flip to every condition and confound the comparison.
    seed_dataset = build_seed_dataset(
        cfg['dataset']['name'],
        cfg['dataset']['root'],
        normalization=cfg['dataset'].get('normalization', 'none'),
        image_size=cfg['dataset'].get('image_size'),
        split=manifest['seed_split'],
    )
    X_src, y_src = source_tensors(manifest, seed_dataset)

    meta = dataset_meta(cfg['dataset']['name'])
    model = build_model(cfg['model']['name'], meta['num_classes'], pretrained=cfg['model'].get('pretrained', True)).to(device)
    load_checkpoint(model, cfg.get('checkpoint'), device, pretrained=cfg['model'].get('pretrained', True))

    before_acc = evaluate(model, test_loader, device)

    X, y, condition_info = build_added_samples(args, cfg, manifest, model, device, X_src, y_src)

    expected_labels = y_src.repeat_interleave(int(manifest['per_source']))
    if not torch.equal(y, expected_labels):
        raise RuntimeError(
            'Added-sample labels do not match the manifest source labels in order; '
            'the condition did not preserve the source protocol.'
        )

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
            'protocol': {
                'sources': args.sources,
                'sources_fingerprint': manifest['fingerprint'],
                'num_sources': int(manifest['num_sources']),
                'per_source': int(manifest['per_source']),
                'source_class_histogram': manifest['class_histogram'],
                'added_class_histogram': class_histogram(y.tolist()),
                'source_transform': 'evaluation (deterministic)',
                'base_training_set': 'full training split with its standard training transform',
            },
            'condition_info': condition_info,
        },
        out_dir / f'controlled_finetune_{args.condition}_run{args.run}.json',
    )


if __name__ == '__main__':
    main()
