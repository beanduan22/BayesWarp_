from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import torch
import torch.nn.functional as F

from bayeswarp.utils.config import load_config
from bayeswarp.utils.device import get_device
from bayeswarp.data.datasets import pixel_range
from bayeswarp.metrics.quality import to_unit_range

BANKS = {
    'mnist': 'results/mnist_lenet4_smoothgrad/failures_suntest_run0.pt',
    'cifar10': 'results/cifar10_vgg16_smoothgrad/failures_suntest_run0.pt',
    'imagenet': 'results/imagenet_resnet50_smoothgrad_quick/failures_suntest_run0.pt',
}


def to_clip_input(x: torch.Tensor, p_min: torch.Tensor, p_max: torch.Tensor) -> torch.Tensor:
    x = to_unit_range(x, p_min, p_max)
    if x.size(1) == 1:
        x = x.repeat(1, 3, 1, 1)
    return F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)


def range_for_bank(bank_path: str):
    cfg = load_config(str(Path('configs') / f'{Path(bank_path).parent.name}.yaml'))
    return pixel_range(cfg['dataset']['name'], cfg['dataset'].get('normalization', 'none'))


def as_4d(x: torch.Tensor) -> torch.Tensor:
    while x.ndim > 4:
        x = x.squeeze(0)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    return x


@torch.no_grad()
def embed(imgs: torch.Tensor, model, device, p_min, p_max, batch: int = 256) -> torch.Tensor:
    out = []
    for i in range(0, imgs.size(0), batch):
        z = model.encode_image(to_clip_input(imgs[i:i + batch], p_min, p_max).to(device))
        out.append(F.normalize(z, dim=-1).cpu())
    return torch.cat(out, dim=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', type=int, default=2000, help='random unrelated pairs per dataset')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cache_dir', default='results/_scs_cache',
                    help='scored-case JSONL cache written by score_all_cases.py')
    ap.add_argument('--out', default='results/_scs_cache/calibration.json')
    args = ap.parse_args()

    import open_clip

    device = get_device()
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model = model.to(device).eval()

    report = {}
    for dataset, bank_path in BANKS.items():
        p = Path(bank_path)
        if not p.exists():
            print(f'{dataset}: {p} missing, skipped')
            continue
        print(f'{dataset}: loading {p}', flush=True)
        pack = torch.load(p, map_location='cpu')
        bank = pack['failure_bank']

        seen: dict[int, torch.Tensor] = {}
        for it in bank:
            s = int(it['seed_idx'])
            if s not in seen:
                seen[s] = as_4d(it['seed_x'].float())
        seeds = sorted(seen)
        imgs = torch.cat([seen[s] for s in seeds], dim=0)
        print(f'  {len(seeds)} distinct seeds', flush=True)
        del pack, bank

        z = embed(imgs, model, device, *range_for_bank(bank_path))

        g = torch.Generator().manual_seed(args.seed)
        n = len(seeds)
        i = torch.randint(0, n, (args.pairs * 2,), generator=g)
        j = torch.randint(0, n, (args.pairs * 2,), generator=g)
        keep = i != j
        i, j = i[keep][:args.pairs], j[keep][:args.pairs]
        sims = (z[i] * z[j]).sum(dim=-1)
        s = sims.sort().values

        floor_median = float(s[len(s) // 2])
        floor_min = float(s[0])
        report[dataset] = {
            'distinct_seeds': n,
            'unrelated_pairs_sampled': int(sims.numel()),
            'floor_min': round(floor_min, 4),
            'floor_p05': round(float(s[int(0.05 * len(s))]), 4),
            'floor_median': round(floor_median, 4),
            'floor_mean': round(float(sims.mean()), 4),
            'floor_p95': round(float(s[int(0.95 * len(s))]), 4),
            'floor_max': round(float(s[-1]), 4),
        }
        print(f'  floor: min={floor_min:.4f} median={floor_median:.4f} '
              f'mean={report[dataset]["floor_mean"]:.4f} max={report[dataset]["floor_max"]:.4f}',
              flush=True)

        cache_rows = []
        for jf in sorted(Path(args.cache_dir).glob('*.jsonl')):
            with open(jf, encoding='utf-8') as fh:
                cache_rows.extend(r for r in (json.loads(x) for x in fh)
                                  if r['dataset'] == dataset)
        if cache_rows:
            gen = sorted(r['scs'] for r in cache_rows)
            below_med = sum(1 for v in gen if v <= floor_median) / len(gen)
            below_min = sum(1 for v in gen if v <= floor_min) / len(gen)
            report[dataset].update({
                'generated_scored': len(gen),
                'generated_min': round(gen[0], 4),
                'generated_median': round(gen[len(gen) // 2], 4),
                'generated_max': round(gen[-1], 4),
                'pct_generated_at_or_below_floor_median': round(100 * below_med, 3),
                'pct_generated_at_or_below_floor_min': round(100 * below_min, 3),
                'headroom_above_floor': round(gen[-1] - floor_median, 4),
                'verdict': ('SCS is not discriminative here: unrelated images already score '
                            f'{floor_median:.3f}, leaving only {gen[-1] - floor_median:.3f} of range '
                            'above the floor')
                           if floor_median > 0.8 else 'floor is low enough for SCS to separate',
            })
            print(f'  generated: min={gen[0]:.4f} median={gen[len(gen)//2]:.4f} max={gen[-1]:.4f}')
            print(f'  {100*below_med:.2f}% of generated cases score at/below the unrelated-pair '
                  f'median ({floor_median:.4f})', flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
