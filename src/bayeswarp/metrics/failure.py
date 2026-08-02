from __future__ import annotations
from statistics import median
from typing import Dict, List


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


def compute_failure_metrics(seed_results: List[Dict], budget: int = 0) -> Dict:
    total_failures = sum(len(r['failures']) for r in seed_results)
    total_seeds = len(seed_results)
    inducing_seed_count = sum(1 for r in seed_results if len(r['failures']) > 0)
    total_time = sum(float(r['time_sec']) for r in seed_results)

    dof_set = set()
    per_seed_dof: List[int] = []
    per_seed_cases: List[int] = []
    qff_values: List[int] = []

    for r in seed_results:
        seed_classes = set()
        for f in r['failures']:
            dof_set.add(int(f['pred']))
            seed_classes.add(int(f['pred']))
        per_seed_dof.append(len(seed_classes))
        per_seed_cases.append(len(r['failures']))
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
