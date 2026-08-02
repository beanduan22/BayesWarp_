from __future__ import annotations
import argparse
import io
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = 'zera09/imagenet_val_full'
SHARDS = 14


def shard_name(i: int) -> str:
    return f'data/validation-{i:05d}-of-{SHARDS:05d}.parquet'


def main():
    parser = argparse.ArgumentParser(
        description='Lay out ImageNet validation images as an ImageFolder tree.\n\n'
                    'Class directories are named by the zero-padded integer label so that '
                    "ImageFolder's alphabetical class ordering reproduces the original label "
                    'ordering exactly; any other naming risks silently permuting the label space.'
    )
    parser.add_argument('--out', default='imagenet/val')
    parser.add_argument('--shards', type=int, default=1,
                        help=f'number of validation shards to extract (1-{SHARDS}, ~3.6k images each)')
    args = parser.parse_args()

    if not 1 <= args.shards <= SHARDS:
        raise SystemExit(f'--shards must be in 1..{SHARDS}')

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    total = 0
    for i in range(args.shards):
        path = hf_hub_download(REPO, shard_name(i), repo_type='dataset')
        table = pq.read_table(path)
        images = table.column('image').to_pylist()
        labels = table.column('label').to_pylist()
        for j, (img, label) in enumerate(zip(images, labels)):
            cls_dir = out / f'{int(label):04d}'
            cls_dir.mkdir(exist_ok=True)
            dst = cls_dir / f'{i:05d}_{j:05d}.JPEG'
            if dst.exists():
                total += 1
                continue
            Image.open(io.BytesIO(img['bytes'])).convert('RGB').save(dst, 'JPEG', quality=95)
            total += 1
        print(f'shard {i}: {len(labels)} images, running total {total}')

    classes = sorted(p.name for p in out.iterdir() if p.is_dir())
    print(f'done: {total} images across {len(classes)} classes under {out}')


if __name__ == '__main__':
    main()
