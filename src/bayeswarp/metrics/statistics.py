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
