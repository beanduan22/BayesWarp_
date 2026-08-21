"""Statistical tests for the paper, aligned by explicit seed identity.

Two things this deliberately does not do:

* it never pairs two vectors by array position. Per-seed metrics are aligned by
  the dataset-level seed id recorded alongside them, and a length mismatch is an
  error rather than a silent truncation to the shorter side;
* it never reports a confidence interval whose estimand disagrees with the
  p-value beside it. Every paired result carries both the mean difference with
  its t interval and t p-value, and the Hodges-Lehmann shift with its
  distribution-free interval and Wilcoxon p-value, each labelled.
"""
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
    hodges_lehmann_ci,
    holm_adjust,
    mean_std,
    paired_difference_ci,
    paired_t_test,
    wilcoxon_signed_rank,
)


# metric -> (values key, seed-id key)
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
    """Mean per seed across runs, keyed by seed id."""
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

    mean, t_low, t_high = paired_difference_ci(a, b)
    t_result = paired_t_test(a, b)
    hl, hl_low, hl_high = hodges_lehmann_ci(a, b)
    w_result = wilcoxon_signed_rank(a, b)

    return {
        'n_pairs': int(a.size),
        'mean_a': float(a.mean()),
        'mean_b': float(b.mean()),
        'median_a': float(np.median(a)),
        'median_b': float(np.median(b)),
        'mean_difference': mean,
        'mean_difference_ci_low': t_low,
        'mean_difference_ci_high': t_high,
        'mean_difference_p_value': t_result['p_value'],
        'mean_difference_statistic': t_result['statistic'],
        'hodges_lehmann_shift': hl,
        'hodges_lehmann_ci_low': hl_low,
        'hodges_lehmann_ci_high': hl_high,
        'wilcoxon_p_value': w_result['p_value'],
        'wilcoxon_statistic': w_result['statistic'],
        'primary_test': 'wilcoxon_signed_rank',
        'primary_estimate': 'hodges_lehmann_shift',
        'estimand_note': (
            'mean_difference_ci pairs with mean_difference_p_value (paired t); '
            'hodges_lehmann_ci pairs with wilcoxon_p_value (signed rank). '
            'The two estimands coincide only for symmetric differences.'
        ),
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
    wilcoxon = wilcoxon_signed_rank(array_a, array_b)
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
        'wilcoxon_p_value': wilcoxon['p_value'],
        'primary_test': 'paired_t',
        'primary_estimate': 'mean_difference',
        'estimand_note': (
            'mean_difference_ci and mean_difference_p_value share the paired-t estimand. '
            'wilcoxon_p_value is reported for reference only and has no interval here.'
        ),
    }


def holm(args):
    values = [float(v) for v in args.p_values]
    return {'mode': 'holm', 'raw': values, 'adjusted': holm_adjust(values)}


def holm_family(args):
    """Holm correction over a family of result files, keeping labels attached.

    Used for the six SCS settings, so the reported significance is family-wise
    corrected rather than six independent tests presented side by side.
    """
    paths = list(args.inputs)
    labels = list(args.labels) if args.labels else [Path(p).stem for p in paths]
    if len(labels) != len(paths):
        raise ValueError('--labels must have one entry per --inputs file')

    # The interval belonging to an estimate shares its prefix, so the family
    # report never pairs an estimate with the other estimand's interval.
    ci_prefix = args.estimate_field[:-len('_shift')] if args.estimate_field.endswith('_shift') else args.estimate_field

    entries = []
    for path, label in zip(paths, labels):
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        if args.p_field not in payload:
            raise KeyError(f'{path} has no "{args.p_field}" field')
        entries.append({
            'label': label,
            'path': path,
            'metric': payload.get('metric'),
            'n_pairs': payload.get('n_pairs'),
            'estimate_field': args.estimate_field,
            'estimate': payload.get(args.estimate_field),
            'ci_low': payload.get(f'{ci_prefix}_ci_low'),
            'ci_high': payload.get(f'{ci_prefix}_ci_high'),
            'p_value': float(payload[args.p_field]),
        })

    adjusted = holm_adjust([e['p_value'] for e in entries])
    for entry, value in zip(entries, adjusted):
        entry['p_adjusted'] = value
        entry['significant_at_0.05'] = bool(value < 0.05)
    return {
        'mode': 'holm_family',
        'family': args.family_name,
        'p_field': args.p_field,
        'estimate_field': args.estimate_field,
        'family_size': len(entries),
        'method': 'holm_bonferroni',
        'results': entries,
        'all_significant_at_0.05': all(e['significant_at_0.05'] for e in entries),
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

    holm_parser = sub.add_parser('holm')
    holm_parser.add_argument('--p-values', nargs='+', required=True)
    holm_parser.add_argument('--out', required=True)
    holm_parser.set_defaults(handler=holm)

    family_parser = sub.add_parser('holm-family')
    family_parser.add_argument('--inputs', nargs='+', required=True)
    family_parser.add_argument('--labels', nargs='*', default=None)
    family_parser.add_argument('--family-name', default='unnamed')
    family_parser.add_argument('--p-field', default='wilcoxon_p_value')
    family_parser.add_argument('--estimate-field', default='hodges_lehmann_shift')
    family_parser.add_argument('--out', required=True)
    family_parser.set_defaults(handler=holm_family)

    kappa_parser = sub.add_parser('kappa')
    kappa_parser.add_argument('--annotations', required=True)
    kappa_parser.add_argument('--out', required=True)
    kappa_parser.set_defaults(handler=kappa)

    args = parser.parse_args()
    save_json(args.handler(args), args.out)


if __name__ == '__main__':
    main()
