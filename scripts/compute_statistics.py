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
    mean_std,
    paired_difference_ci,
    paired_t_test,
)


SEED_KEYS = {
    'nof': ('per_seed_cases', 'seed_ids'),
    'dof': ('per_seed_dof', 'seed_ids'),
    'qff': ('per_seed_qff', 'seed_ids'),
    'scs': ('per_seed_scs', 'scs_seed_ids'),
}


def load_runs(pattern: str):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f'No files matched: {pattern}')
    return [(p, json.loads(Path(p).read_text(encoding='utf-8'))) for p in paths]


def seed_series(path: str, run: dict, value_key: str, id_key: str):
    if value_key not in run:
        raise KeyError(f'{path} has no "{value_key}" field')
    values = [float(v) for v in run[value_key]]
    if id_key not in run:
        raise KeyError(
            f'{path} has no "{id_key}" field, so per-seed values cannot be paired by identity. '
            'Re-generate this result with the current pipeline, which records seed ids.'
        )
    ids = [int(v) for v in run[id_key]]
    if len(ids) != len(values):
        raise ValueError(f'{path}: {len(ids)} seed ids for {len(values)} values of "{value_key}"')
    if len(set(ids)) != len(ids):
        raise ValueError(f'{path}: duplicate seed ids in "{id_key}"')
    return dict(zip(ids, values))


def average_across_runs(runs, value_key: str, id_key: str, allow_intersection: bool):
    series = [seed_series(path, run, value_key, id_key) for path, run in runs]
    id_sets = [set(s) for s in series]
    shared = set.intersection(*id_sets)
    union = set.union(*id_sets)
    if shared != union and not allow_intersection:
        missing = sorted(union - shared)[:10]
        raise ValueError(
            f'Runs do not cover the same seeds ({len(union)} distinct ids, {len(shared)} shared; '
            f'e.g. {missing}). Pass --allow-seed-intersection to restrict to shared seeds and '
            'record the drop in the output.'
        )
    ids = sorted(shared)
    values = {i: float(np.mean([s[i] for s in series])) for i in ids}
    return values, {'n_runs': len(series), 'ids_union': len(union), 'ids_used': len(ids)}


def paired_result(a_values, b_values, ids):
    a = np.array([a_values[i] for i in ids], dtype=float)
    b = np.array([b_values[i] for i in ids], dtype=float)

    mean, low, high = paired_difference_ci(a, b)
    t_result = paired_t_test(a, b)

    return {
        'n_pairs': int(a.size),
        'mean_a': float(a.mean()),
        'mean_b': float(b.mean()),
        'median_a': float(np.median(a)),
        'median_b': float(np.median(b)),
        'mean_difference': mean,
        'mean_difference_ci_low': low,
        'mean_difference_ci_high': high,
        'mean_difference_p_value': t_result['p_value'],
        'mean_difference_statistic': t_result['statistic'],
        'test': 'paired_t_two_sided',
        'estimate': 'mean_difference',
        'confidence_level': 0.95,
        'multiplicity_adjustment': 'none',
    }


def paired_seed(args):
    value_key, id_key = SEED_KEYS[args.metric]
    a_values, a_info = average_across_runs(
        load_runs(args.a), value_key, id_key, args.allow_seed_intersection
    )
    b_values, b_info = average_across_runs(
        load_runs(args.b), value_key, id_key, args.allow_seed_intersection
    )

    a_ids, b_ids = set(a_values), set(b_values)
    shared = sorted(a_ids & b_ids)
    if a_ids != b_ids and not args.allow_seed_intersection:
        only_a = sorted(a_ids - b_ids)[:10]
        only_b = sorted(b_ids - a_ids)[:10]
        raise ValueError(
            f'Condition A covers {len(a_ids)} seeds and condition B covers {len(b_ids)}; '
            f'{len(shared)} are shared. Only in A: {only_a}. Only in B: {only_b}. '
            'Pass --allow-seed-intersection to compare the shared seeds and record the drop.'
        )
    if not shared:
        raise ValueError('No seed ids are shared between the two conditions')

    result = {
        'mode': 'paired_seed',
        'metric': args.metric,
        'value_key': value_key,
        'id_key': id_key,
        'pairing': 'by seed id',
        'a_pattern': args.a,
        'b_pattern': args.b,
        'a_runs': a_info,
        'b_runs': b_info,
        'seeds_only_in_a': len(a_ids - b_ids),
        'seeds_only_in_b': len(b_ids - a_ids),
        'seed_ids': shared,
    }
    result.update(paired_result(a_values, b_values, shared))
    return result


def paired_run(args):
    a_runs = load_runs(args.a)
    b_runs = load_runs(args.b)
    a = [float(run[args.metric]) for _, run in a_runs]
    b = [float(run[args.metric]) for _, run in b_runs]
    if len(a) != len(b):
        raise ValueError(
            f'Condition A has {len(a)} runs and condition B has {len(b)}; '
            'paired run-level tests need the same runs on both sides.'
        )
    a_ids = [int(run.get('run', i)) for i, (_, run) in enumerate(a_runs)]
    b_ids = [int(run.get('run', i)) for i, (_, run) in enumerate(b_runs)]
    if a_ids != b_ids:
        raise ValueError(f'Run ids differ between conditions: {a_ids} vs {b_ids}')

    array_a, array_b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mean, low, high = paired_difference_ci(array_a, array_b)
    result = paired_t_test(array_a, array_b)
    mean_a, std_a = mean_std(a)
    mean_b, std_b = mean_std(b)
    return {
        'mode': 'paired_run',
        'metric': args.metric,
        'pairing': 'by run id',
        'run_ids': a_ids,
        'n_runs': len(a),
        'mean_a': mean_a,
        'std_a': std_a,
        'mean_b': mean_b,
        'std_b': std_b,
        'mean_difference': mean,
        'mean_difference_ci_low': low,
        'mean_difference_ci_high': high,
        'mean_difference_p_value': result['p_value'],
        'mean_difference_statistic': result['statistic'],
        'test': 'paired_t_two_sided',
        'estimate': 'mean_difference',
        'confidence_level': 0.95,
        'multiplicity_adjustment': 'none',
    }


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
    seed_parser.add_argument(
        '--allow-seed-intersection', action='store_true',
        help='Compare only the seeds present on both sides instead of failing on a mismatch. '
             'The number dropped is recorded in the output.',
    )
    seed_parser.add_argument('--out', required=True)
    seed_parser.set_defaults(handler=paired_seed)

    run_parser = sub.add_parser('paired-run')
    run_parser.add_argument('--a', required=True)
    run_parser.add_argument('--b', required=True)
    run_parser.add_argument('--metric', required=True)
    run_parser.add_argument('--out', required=True)
    run_parser.set_defaults(handler=paired_run)

    kappa_parser = sub.add_parser('kappa')
    kappa_parser.add_argument('--annotations', required=True)
    kappa_parser.add_argument('--out', required=True)
    kappa_parser.set_defaults(handler=kappa)

    args = parser.parse_args()
    save_json(args.handler(args), args.out)


if __name__ == '__main__':
    main()
