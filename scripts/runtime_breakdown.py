from __future__ import annotations
import argparse
import copy
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from bayeswarp.pipeline import generate
from bayeswarp.testing.bayeswarp import STAGES
from bayeswarp.utils.config import load_config
from bayeswarp.utils.device import get_device
from bayeswarp.utils.io import ensure_dir, save_json


def summarize(metrics, num_seeds: int):
    stages = metrics.get('stage_time_sec', {})
    total = metrics.get('total_time_sec', 0.0)
    accounted = sum(stages.get(stage, 0.0) for stage in STAGES)
    return {
        'total_time_sec': total,
        'time_per_seed_sec': total / max(1, num_seeds),
        'stage_time_sec': {stage: stages.get(stage, 0.0) for stage in STAGES},
        'unaccounted_time_sec': total - accounted,
    }


def sweep(base_cfg, device, key, values, num_seeds, run, label):
    entries = []
    for value in values:
        cfg = copy.deepcopy(base_cfg)
        cfg['bayeswarp'][key] = value
        _, _, metrics = generate(cfg, device, run=run, num_seeds=num_seeds, progress=f'{label}={value}')
        entry = summarize(metrics, num_seeds or cfg['test']['num_seeds'])
        entry[key] = value
        entries.append(entry)
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--num-seeds', type=int, default=10)
    parser.add_argument('--budgets', type=int, nargs='+', default=[500, 1000, 2000, 4000])
    parser.add_argument('--inducing-points', type=int, nargs='+', default=[32, 64, 128, 256])
    parser.add_argument('--target-classes', type=int, nargs='+', default=[1, 3, 5, 9])
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    device = get_device()
    out_dir = ensure_dir(base_cfg['output_dir'])

    _, _, metrics = generate(base_cfg, device, run=args.run, num_seeds=args.num_seeds, progress='Runtime breakdown')

    payload = {
        'breakdown': summarize(metrics, args.num_seeds),
        'budget_scaling': sweep(base_cfg, device, 'budget', args.budgets, args.num_seeds, args.run, 'budget'),
        'inducing_point_scaling': sweep(base_cfg, device, 'm', args.inducing_points, args.num_seeds, args.run, 'm'),
        'target_class_scaling': sweep(
            base_cfg, device, 'max_target_classes', args.target_classes, args.num_seeds, args.run, 'K'
        ),
    }

    save_json(payload, out_dir / f'runtime_breakdown_run{args.run}.json')


if __name__ == '__main__':
    main()
