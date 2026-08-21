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


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.allclose(a, b):
        return {'statistic': float('nan'), 'p_value': 1.0}
    result = stats.wilcoxon(a, b, alternative='two-sided', zero_method='wilcox')
    return {'statistic': float(result.statistic), 'p_value': float(result.pvalue)}


def paired_t_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    result = stats.ttest_rel(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return {'statistic': float(result.statistic), 'p_value': float(result.pvalue)}


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    values = list(p_values)
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    adjusted = [0.0] * n
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (n - rank) * values[index]
        running = max(running, min(candidate, 1.0))
        adjusted[index] = running
    return adjusted


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


def _paired_differences(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if diff.ndim != 1:
        raise ValueError('paired inputs must be one-dimensional')
    return diff


def _walsh_averages(diff: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(diff.size, k=0)
    return np.sort((diff[i] + diff[j]) / 2.0)


def hodges_lehmann(a: Sequence[float], b: Sequence[float]) -> float:
    """Median of the Walsh averages of the paired differences.

    This is the location estimate whose null hypothesis the Wilcoxon signed-rank
    test evaluates, so it is the estimate that belongs next to a Wilcoxon p-value.
    """
    diff = _paired_differences(a, b)
    if diff.size == 0:
        return float('nan')
    return float(np.median(_walsh_averages(diff)))


def hodges_lehmann_ci(
    a: Sequence[float],
    b: Sequence[float],
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Distribution-free CI for the Hodges-Lehmann estimate of the paired shift.

    Uses the normal approximation to the signed-rank distribution with a
    continuity correction to pick the order statistics of the Walsh averages.
    Returns ``(estimate, low, high)``.
    """
    diff = _paired_differences(a, b)
    n = diff.size
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    walsh = _walsh_averages(diff)
    estimate = float(np.median(walsh))
    total = walsh.size
    if n < 3:
        return estimate, float('nan'), float('nan')
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    k = int(math.floor(mu - z * sigma - 0.5))
    if k < 0:
        return estimate, float(walsh[0]), float(walsh[-1])
    k = min(k, total // 2 - 1) if total >= 2 else 0
    if k < 0:
        return estimate, float(walsh[0]), float(walsh[-1])
    return estimate, float(walsh[k]), float(walsh[total - 1 - k])


def cluster_bootstrap_interval(
    labels: Sequence[int],
    clusters: Sequence,
    confidence: float = 0.95,
    iterations: int = 10000,
    seed: int = 0,
) -> Dict[str, float]:
    """Percentile bootstrap for a proportion when observations share clusters.

    Clusters (for example, the source seed a generated case came from) are
    resampled with replacement; every observation inside a resampled cluster is
    carried along. This keeps the interval honest when the same source seed
    contributes several annotated cases, which a Wilson interval would ignore.
    """
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
