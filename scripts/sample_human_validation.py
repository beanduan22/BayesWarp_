from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import torch
from torchvision.utils import save_image

from bayeswarp.utils.config import load_config
from bayeswarp.utils.io import ensure_dir, save_json


def denormalize(x: torch.Tensor, normalization: str) -> torch.Tensor:
    image = x if x.ndim == 3 else x.squeeze(0)
    if normalization == 'imagenet' and image.size(0) == 3:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
    return image.clamp(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--failures', nargs='+', required=True)
    parser.add_argument('--methods', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--per-method', type=int, default=30)
    parser.add_argument('--class-meta', default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if len(args.failures) != len(args.methods):
        raise ValueError('--failures and --methods must have the same length')

    cfg = load_config(args.config)
    normalization = cfg['dataset'].get('normalization', 'none')
    class_meta = json.loads(Path(args.class_meta).read_text(encoding='utf-8')) if args.class_meta else {}

    out_dir = ensure_dir(args.out_dir)
    image_dir = ensure_dir(Path(out_dir) / 'images')
    rng = random.Random(args.seed)

    tasks = []
    key = {}
    for failures_path, method in zip(args.failures, args.methods):
        pack = torch.load(failures_path, map_location='cpu')
        bank = pack['failure_bank']
        if not bank:
            raise RuntimeError(f'No generated cases in {failures_path}')
        sample = bank if len(bank) <= args.per_method else rng.sample(bank, args.per_method)
        for item in sample:
            task_id = f'{cfg["model"]["name"]}_{len(tasks):04d}'
            original = denormalize(item['seed_x'], normalization)
            generated = denormalize(item['x'], normalization)
            save_image(original, Path(image_dir) / f'{task_id}_original.png')
            save_image(generated, Path(image_dir) / f'{task_id}_generated.png')

            entry = {
                'task_id': task_id,
                'model': cfg['model']['name'],
                'dataset': cfg['dataset']['name'],
                'original_image': f'images/{task_id}_original.png',
                'generated_image': f'images/{task_id}_generated.png',
                'question': 'Does the generated image still show the same object class as the original image?',
            }
            source_class = str(int(item['seed_y']))
            if source_class in class_meta:
                entry['source_class_name'] = class_meta[source_class].get('name')
                entry['source_class_synonyms'] = class_meta[source_class].get('synonyms')
                entry['source_class_definition'] = class_meta[source_class].get('definition')
            tasks.append(entry)

            key[task_id] = {
                'method': method,
                'model': cfg['model']['name'],
                'dataset': cfg['dataset']['name'],
                'seed_idx': int(item['seed_idx']),
                'source_label': int(item['seed_y']),
                'original_prediction': int(item['og']),
                'model_prediction': int(item['pred']),
            }

    rng.shuffle(tasks)
    save_json(tasks, Path(out_dir) / 'tasks.json')
    save_json(key, Path(out_dir) / 'key.json')


if __name__ == '__main__':
    main()
