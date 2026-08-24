from __future__ import annotations
from typing import Dict, List, Sequence, Tuple

import math

import numpy as np
from scipy import stats


def paired_difference_ci(a: Sequence[float], b: Sequence[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = diff.size
    mean = float(diff.mean())
    if n < 2:
        return mean, float('nan'), float('nan')
    sem = float(diff.std(ddof=1) / math.sqrt(n))
    half = float(stats.t.ppf(0.5 + confidence / 2.0, df=n - 1) * sem)
    return mean, mean - half, mean + half


def paired_t_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    result = stats.ttest_rel(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return {'statistic': float(result.statistic), 'p_value': float(result.pvalue)}


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    if total == 0:
        return float('nan'), float('nan'), float('nan')
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    phat = successes / total
    denominator = 1.0 + z * z / total
    center = (phat + z * z / (2 * total)) / denominator
    half = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denominator
    return phat, max(0.0, center - half), min(1.0, center + half)


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    a = np.asarray(list(a))
    b = np.asarray(list(b))
    labels = sorted(set(a.tolist()) | set(b.tolist()))
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=float)
    for x, y in zip(a, b):
        matrix[index[x], index[y]] += 1.0
    total = matrix.sum()
    if total == 0:
        return float('nan')
    observed = np.trace(matrix) / total
    expected = float((matrix.sum(axis=0) * matrix.sum(axis=1)).sum()) / (total * total)
    if abs(1.0 - expected) < 1e-12:
        return float('nan')
    return float((observed - expected) / (1.0 - expected))


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return float('nan'), float('nan')
    if array.size == 1:
        return float(array.mean()), 0.0
    return float(array.mean()), float(array.std(ddof=1))


def median_iqr(values: Sequence[float]) -> Tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return float('nan'), float('nan'), float('nan')
    return (
        float(np.median(array)),
        float(np.percentile(array, 25)),
        float(np.percentile(array, 75)),
    )


def cluster_bootstrap_interval(
    labels: Sequence[int],
    clusters: Sequence,
    confidence: float = 0.95,
    iterations: int = 10000,
    seed: int = 0,
) -> Dict[str, float]:
    values = np.asarray(list(labels), dtype=float)
    keys = list(clusters)
    if values.size != len(keys):
        raise ValueError('labels and clusters must have the same length')
    if values.size == 0:
        return {
            'point': float('nan'),
            'ci_low': float('nan'),
            'ci_high': float('nan'),
            'n_clusters': 0,
            'max_cluster_size': 0,
            'iterations': 0,
        }

    grouped: Dict[object, List[float]] = {}
    for value, key in zip(values.tolist(), keys):
        grouped.setdefault(key, []).append(value)
    group_keys = sorted(grouped, key=lambda k: str(k))
    sums = np.array([sum(grouped[k]) for k in group_keys], dtype=float)
    sizes = np.array([len(grouped[k]) for k in group_keys], dtype=float)
    n_clusters = len(group_keys)
    point = float(values.mean())

    if n_clusters < 2:
        return {
            'point': point,
            'ci_low': float('nan'),
            'ci_high': float('nan'),
            'n_clusters': n_clusters,
            'max_cluster_size': int(sizes.max()) if n_clusters else 0,
            'iterations': 0,
        }

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_clusters, size=(iterations, n_clusters))
    boot = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(boot, [alpha, 1.0 - alpha])
    return {
        'point': point,
        'ci_low': float(low),
        'ci_high': float(high),
        'n_clusters': int(n_clusters),
        'max_cluster_size': int(sizes.max()),
        'iterations': int(iterations),
    }
