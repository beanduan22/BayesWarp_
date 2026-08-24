from __future__ import annotations
from statistics import median
from typing import Dict, List, Optional, Sequence


def _quartiles(values: List[float]):
    if not values:
        return None, None, None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    lower = ordered[:mid]
    upper = ordered[mid + 1:] if n % 2 else ordered[mid:]
    q1 = median(lower) if lower else ordered[0]
    q3 = median(upper) if upper else ordered[-1]
    return float(median(ordered)), float(q1), float(q3)


def compute_failure_metrics(
    seed_results: List[Dict],
    budget: int = 0,
    seed_ids: Optional[Sequence[int]] = None,
) -> Dict:
    if seed_ids is None:
        resolved_ids = [
            int(r.get('seed_id', r.get('seed_idx', position)))
            for position, r in enumerate(seed_results)
        ]
    else:
        resolved_ids = [int(v) for v in seed_ids]
        if len(resolved_ids) != len(seed_results):
            raise ValueError('seed_ids must have one entry per seed result')
    if len(set(resolved_ids)) != len(resolved_ids):
        raise ValueError('seed ids must be unique so per-seed metrics can be paired by id')

    counts = [int(r.get('n_failures', len(r['failures']))) for r in seed_results]
    preds = [
        [int(v) for v in r['failure_preds']] if 'failure_preds' in r
        else [int(f['pred']) for f in r['failures']]
        for r in seed_results
    ]

    total_failures = sum(counts)
    total_seeds = len(seed_results)
    inducing_seed_count = sum(1 for c in counts if c > 0)
    total_time = sum(float(r['time_sec']) for r in seed_results)

    dof_set = set()
    per_seed_dof: List[int] = []
    per_seed_cases: List[int] = []
    qff_values: List[int] = []

    for r, count, seed_preds in zip(seed_results, counts, preds):
        dof_set.update(seed_preds)
        per_seed_dof.append(len(set(seed_preds)))
        per_seed_cases.append(count)
        qff_values.append(int(r.get('qff', budget)))

    successful_cases = [c for c in per_seed_cases if c > 0]
    successful_dof = [d for d, c in zip(per_seed_dof, per_seed_cases) if c > 0]

    cases_med, cases_q1, cases_q3 = _quartiles([float(v) for v in successful_cases])
    classes_med, classes_q1, classes_q3 = _quartiles([float(v) for v in successful_dof])
    qff_med, qff_q1, qff_q3 = _quartiles([float(v) for v in qff_values])

    stage_totals: Dict[str, float] = {}
    for r in seed_results:
        for stage, value in r.get('stage_time_sec', {}).items():
            stage_totals[stage] = stage_totals.get(stage, 0.0) + float(value)

    return {
        'seed_ids': resolved_ids,
        'stored_cases': int(sum(len(r['failures']) for r in seed_results)),
        'NoF': int(total_failures),
        'FSR': float(inducing_seed_count / max(1, total_seeds)),
        'TPF': float(total_time / max(1, total_failures)),
        'DoF': int(len(dof_set)),
        'QFF_median': qff_med,
        'QFF_q1': qff_q1,
        'QFF_q3': qff_q3,
        'QFF_mean': float(sum(qff_values) / max(1, len(qff_values))),
        'cases_per_successful_seed_median': cases_med,
        'cases_per_successful_seed_q1': cases_q1,
        'cases_per_successful_seed_q3': cases_q3,
        'classes_per_successful_seed_median': classes_med,
        'classes_per_successful_seed_q1': classes_q1,
        'classes_per_successful_seed_q3': classes_q3,
        'per_seed_dof': per_seed_dof,
        'per_seed_cases': per_seed_cases,
        'per_seed_qff': qff_values,
        'total_time_sec': float(total_time),
        'stage_time_sec': stage_totals,
    }
