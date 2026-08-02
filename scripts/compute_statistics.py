from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np

from bayeswarp.utils.io import save_json
from bayeswarp.metrics.statistics import (
    cohen_kappa,
    holm_adjust,
    mean_std,
    paired_difference_ci,
    paired_t_test,
    wilcoxon_signed_rank,
)


SEED_KEYS = {
    'nof': 'per_seed_cases',
    'dof': 'per_seed_dof',
    'qff': 'per_seed_qff',
}


def load_runs(pattern: str):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f'No files matched: {pattern}')
    return [json.loads(Path(p).read_text(encoding='utf-8')) for p in paths]


def average_per_seed(runs, key: str) -> np.ndarray:
    arrays = [np.asarray(run[key], dtype=float) for run in runs]
    length = min(a.size for a in arrays)
    stacked = np.stack([a[:length] for a in arrays], axis=0)
    return stacked.mean(axis=0)


def paired_seed(args):
    key = SEED_KEYS[args.metric]
    a = average_per_seed(load_runs(args.a), key)
    b = average_per_seed(load_runs(args.b), key)
    length = min(a.size, b.size)
    a, b = a[:length], b[:length]

    mean, low, high = paired_difference_ci(a, b)
    result = wilcoxon_signed_rank(a, b)
    return {
        'mode': 'paired_seed',
        'metric': args.metric,
        'n_pairs': int(length),
        'mean_a': float(a.mean()),
        'mean_b': float(b.mean()),
        'mean_difference': mean,
        'ci_low': low,
        'ci_high': high,
        'statistic': result['statistic'],
        'p_value': result['p_value'],
    }


def paired_run(args):
    a_runs = load_runs(args.a)
    b_runs = load_runs(args.b)
    a = [float(run[args.metric]) for run in a_runs]
    b = [float(run[args.metric]) for run in b_runs]
    length = min(len(a), len(b))
    a, b = a[:length], b[:length]

    mean, low, high = paired_difference_ci(a, b)
    result = paired_t_test(a, b)
    mean_a, std_a = mean_std(a)
    mean_b, std_b = mean_std(b)
    return {
        'mode': 'paired_run',
        'metric': args.metric,
        'n_runs': int(length),
        'mean_a': mean_a,
        'std_a': std_a,
        'mean_b': mean_b,
        'std_b': std_b,
        'mean_difference': mean,
        'ci_low': low,
        'ci_high': high,
        'statistic': result['statistic'],
        'p_value': result['p_value'],
    }


def holm(args):
    values = [float(v) for v in args.p_values]
    return {'mode': 'holm', 'raw': values, 'adjusted': holm_adjust(values)}


def kappa(args):
    payload = json.loads(Path(args.annotations).read_text(encoding='utf-8'))
    first = [int(item['annotator_a']) for item in payload]
    second = [int(item['annotator_b']) for item in payload]
    return {'mode': 'kappa', 'n': len(first), 'cohen_kappa': cohen_kappa(first, second)}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)

    seed_parser = sub.add_parser('paired-seed')
    seed_parser.add_argument('--a', required=True)
    seed_parser.add_argument('--b', required=True)
    seed_parser.add_argument('--metric', required=True, choices=sorted(SEED_KEYS))
    seed_parser.add_argument('--out', required=True)
    seed_parser.set_defaults(handler=paired_seed)

    run_parser = sub.add_parser('paired-run')
    run_parser.add_argument('--a', required=True)
    run_parser.add_argument('--b', required=True)
    run_parser.add_argument('--metric', required=True)
    run_parser.add_argument('--out', required=True)
    run_parser.set_defaults(handler=paired_run)

    holm_parser = sub.add_parser('holm')
    holm_parser.add_argument('--p-values', nargs='+', required=True)
    holm_parser.add_argument('--out', required=True)
    holm_parser.set_defaults(handler=holm)

    kappa_parser = sub.add_parser('kappa')
    kappa_parser.add_argument('--annotations', required=True)
    kappa_parser.add_argument('--out', required=True)
    kappa_parser.set_defaults(handler=kappa)

    args = parser.parse_args()
    save_json(args.handler(args), args.out)


if __name__ == '__main__':
    main()
