from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import torch
from torchvision.utils import save_image

from bayeswarp.utils.config import load_config
from bayeswarp.utils.device import get_device
from bayeswarp.utils.io import ensure_dir, save_json
from bayeswarp.finetune.sources import bank_seed_ids
from bayeswarp.pipeline import build_seeds, build_subject


def denormalize(x: torch.Tensor, normalization: str) -> torch.Tensor:
    image = x if x.ndim == 3 else x.squeeze(0)
    if normalization == 'imagenet' and image.size(0) == 3:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
    return image.clamp(0, 1)


def random_sample(candidates, per_method: int, rng: random.Random):
    if len(candidates) <= per_method:
        return list(candidates)
    return rng.sample(candidates, per_method)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--failures', nargs='+', required=True)
    parser.add_argument('--methods', nargs='+', required=True)
    parser.add_argument(
        '--runs', nargs='*', default=None,
        help='Run id per --failures pack. Defaults to 0 for every pack. Supply several packs '
             'per method to spread the sample across runs.',
    )
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--per-method', type=int, default=30)
    parser.add_argument('--class-meta', default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if len(args.failures) != len(args.methods):
        raise ValueError('--failures and --methods must have the same length')
    runs = [int(v) for v in args.runs] if args.runs else [0] * len(args.failures)
    if len(runs) != len(args.failures):
        raise ValueError('--runs must have one entry per --failures pack')

    cfg = load_config(args.config)
    normalization = cfg['dataset'].get('normalization', 'none')
    class_meta = json.loads(Path(args.class_meta).read_text(encoding='utf-8')) if args.class_meta else {}

    out_dir = ensure_dir(args.out_dir)
    image_dir = ensure_dir(Path(out_dir) / 'images')
    rng = random.Random(args.seed)

    fallback = None
    pools = defaultdict(list)
    for failures_path, method, run in zip(args.failures, args.methods, runs):
        pack = torch.load(failures_path, map_location='cpu')
        bank = pack['failure_bank']
        if not bank:
            raise RuntimeError(f'No generated cases in {failures_path}')
        seed_results = pack.get('seed_results') or []
        if not (seed_results and 'seed_id' in seed_results[0]) and fallback is None:
            device = get_device()
            fallback = [int(i) for i in build_seeds(cfg, build_subject(cfg, device), device).indices]
        ids = bank_seed_ids(pack, fallback)
        for item, seed_id in zip(bank, ids):
            pools[method].append({
                'item': item,
                'seed_id': int(seed_id),
                'source_label': int(item['seed_y']),
                'run': int(run),
                'source_pack': failures_path,
            })
        del pack

    tasks = []
    key = {}
    coverage = {}
    for method in dict.fromkeys(args.methods):
        candidates = pools[method]
        selected = random_sample(candidates, args.per_method, rng)
        if len(selected) < args.per_method:
            raise RuntimeError(
                f'Method {method} could only supply {len(selected)} of {args.per_method} cases; '
                f'its pool holds {len(candidates)}. Add more runs.'
            )
        seed_counts = defaultdict(int)
        run_counts = defaultdict(int)
        class_counts = defaultdict(int)
        for entry in selected:
            item = entry['item']
            task_id = f'{cfg["model"]["name"]}_{len(tasks):04d}'
            original = denormalize(item['seed_x'], normalization)
            generated = denormalize(item['x'], normalization)
            save_image(original, Path(image_dir) / f'{task_id}_original.png')
            save_image(generated, Path(image_dir) / f'{task_id}_generated.png')

            entry_json = {
                'task_id': task_id,
                'model': cfg['model']['name'],
                'dataset': cfg['dataset']['name'],
                'original_image': f'images/{task_id}_original.png',
                'generated_image': f'images/{task_id}_generated.png',
                'question': 'Does the generated image still show the same object class as the original image?',
            }
            source_class = str(entry['source_label'])
            if source_class in class_meta:
                entry_json['source_class_name'] = class_meta[source_class].get('name')
                entry_json['source_class_synonyms'] = class_meta[source_class].get('synonyms')
                entry_json['source_class_definition'] = class_meta[source_class].get('definition')
            tasks.append(entry_json)

            key[task_id] = {
                'method': method,
                'model': cfg['model']['name'],
                'dataset': cfg['dataset']['name'],
                'run': entry['run'],
                'source_pack': entry['source_pack'],
                'seed_id': entry['seed_id'],
                'seed_idx': int(item['seed_idx']),
                'source_label': entry['source_label'],
                'original_prediction': int(item['og']),
                'model_prediction': int(item['pred']),
            }
            seed_counts[entry['seed_id']] += 1
            run_counts[entry['run']] += 1
            class_counts[entry['source_label']] += 1

        coverage[method] = {
            'sampled': len(selected),
            'pool_cases': len(candidates),
            'pool_seeds': len({e['seed_id'] for e in candidates}),
            'distinct_seeds': len(seed_counts),
            'max_cases_per_seed': max(seed_counts.values()),
            'runs': {str(k): v for k, v in sorted(run_counts.items())},
            'source_classes': len(class_counts),
            'max_cases_per_class': max(class_counts.values()),
        }

    rng.shuffle(tasks)
    save_json(tasks, Path(out_dir) / 'tasks.json')
    save_json(key, Path(out_dir) / 'key.json')
    save_json(
        {
            'per_method': args.per_method,
            'sampling': 'uniform random without replacement from the pooled cases per method',
            'sampling_seed': args.seed,
            'packs': [
                {'method': m, 'run': r, 'path': p}
                for p, m, r in zip(args.failures, args.methods, runs)
            ],
            'coverage': coverage,
        },
        Path(out_dir) / 'sampling.json',
    )


if __name__ == '__main__':
    main()
