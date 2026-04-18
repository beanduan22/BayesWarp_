from __future__ import annotations
from typing import Dict, List


def compute_failure_metrics(seed_results: List[Dict]):
    total_failures = sum(len(r['failures']) for r in seed_results)
    total_seeds = len(seed_results)
    inducing_seed_count = sum(1 for r in seed_results if len(r['failures']) > 0)
    total_time = sum(float(r['time_sec']) for r in seed_results)
    dof_set = set()
    for r in seed_results:
        for f in r['failures']:
            dof_set.add(int(f['pred']))
    return {
        'NoF': int(total_failures),
        'FSR': float(inducing_seed_count / max(1, total_seeds)),
        'TPF': float(total_time / max(1, total_failures)),
        'DoF': int(len(dof_set)),
    }
