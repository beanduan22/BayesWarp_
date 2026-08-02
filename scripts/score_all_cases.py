from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import torch
import torch.nn.functional as F

from bayeswarp.utils.device import get_device
from bayeswarp.utils.io import ensure_dir

DATASETS = {
    'mnist': ['results/mnist_lenet4_smoothgrad', 'results/mnist_lenet5_smoothgrad'],
    'cifar10': ['results/cifar10_vgg16_smoothgrad', 'results/cifar10_resnet18_smoothgrad'],
    'imagenet': ['results/imagenet_resnet50_smoothgrad_quick'],
}
METHODS = ('adapt', 'nsgen', 'suntest')


def to_clip_input(x: torch.Tensor) -> torch.Tensor:
    if x.size(1) == 1:
        x = x.repeat(1, 3, 1, 1)
    x = x.clamp(0, 1)
    return F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)


def as_4d(x: torch.Tensor) -> torch.Tensor:
    while x.ndim > 4:
        x = x.squeeze(0)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    return x


@torch.no_grad()
def score_bank(bank, model, device, batch_size: int):
    scores = []
    for start in range(0, len(bank), batch_size):
        chunk = bank[start:start + batch_size]
        seeds = torch.cat([as_4d(it['seed_x'].float()) for it in chunk], dim=0)
        gens = torch.cat([as_4d(it['x'].float()) for it in chunk], dim=0)
        z1 = model.encode_image(to_clip_input(seeds).to(device))
        z2 = model.encode_image(to_clip_input(gens).to(device))
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        scores.extend((z1 * z2).sum(dim=-1).float().cpu().tolist())
        print(f'    {min(start + batch_size, len(bank))}/{len(bank)}', end='\r', flush=True)
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='results/_scs_cache')
    ap.add_argument('--batch_size', type=int, default=256)
    args = ap.parse_args()

    import open_clip

    device = get_device()
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model = model.to(device).eval()
    cache = ensure_dir(args.cache)

    for dataset, dirs in DATASETS.items():
        for d in dirs:
            for method in METHODS:
                p = Path(d) / f'failures_{method}.pt'
                if not p.exists():
                    continue
                model_name = Path(d).name
                out = Path(cache) / f'{model_name}__{method}.jsonl'
                if out.exists():
                    print(f'skip {out} (exists)')
                    continue

                print(f'loading {p} ...', flush=True)
                pack = torch.load(p, map_location='cpu')
                bank = pack['failure_bank']
                print(f'  {len(bank)} cases; scoring', flush=True)
                scores = score_bank(bank, model, device, args.batch_size)

                with open(out, 'w', encoding='utf-8') as f:
                    for i, (it, s) in enumerate(zip(bank, scores)):
                        f.write(json.dumps({
                            'case_id': f'{model_name}__{method}#{i}',
                            'dataset': dataset,
                            'model': model_name,
                            'method': method,
                            'index': i,
                            'seed_idx': int(it['seed_idx']),
                            'seed_y': int(it['seed_y']),
                            'og': int(it['og']),
                            'pred': int(it['pred']),
                            'scs': float(s),
                        }) + '\n')
                del pack, bank
                print(f'  wrote {out}', flush=True)

    print('SCORING DONE', flush=True)


if __name__ == '__main__':
    main()
