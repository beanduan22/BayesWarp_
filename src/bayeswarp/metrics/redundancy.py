from __future__ import annotations
from typing import List, Sequence

import torch


def normalized_l2(vectors: torch.Tensor) -> torch.Tensor:
    return vectors.flatten(start_dim=1) / (vectors[0].numel() ** 0.5)


def pairwise_distance_quantile(vectors: torch.Tensor, quantile: float, max_samples: int = 2048) -> float:
    flat = normalized_l2(vectors)
    if flat.size(0) > max_samples:
        index = torch.randperm(flat.size(0))[:max_samples]
        flat = flat[index]
    if flat.size(0) < 2:
        return 0.0
    distances = torch.pdist(flat)
    return float(torch.quantile(distances, quantile).item())


def count_distinct(vectors: torch.Tensor, threshold: float) -> int:
    flat = normalized_l2(vectors)
    representatives: List[torch.Tensor] = []
    for i in range(flat.size(0)):
        candidate = flat[i]
        assigned = False
        for representative in representatives:
            if float(torch.norm(candidate - representative)) <= threshold:
                assigned = True
                break
        if not assigned:
            representatives.append(candidate)
    return len(representatives)


def redundancy_summary(
    pixel_vectors: torch.Tensor,
    feature_vectors: torch.Tensor,
    pixel_threshold: float,
    feature_threshold: float,
) -> dict:
    total = int(pixel_vectors.size(0))
    distinct_pixel = count_distinct(pixel_vectors, pixel_threshold)
    distinct_feature = count_distinct(feature_vectors, feature_threshold)
    return {
        'total_cases': total,
        'distinct_pixel': distinct_pixel,
        'distinct_pixel_ratio': distinct_pixel / max(1, total),
        'distinct_feature': distinct_feature,
        'distinct_feature_ratio': distinct_feature / max(1, total),
        'pixel_threshold': float(pixel_threshold),
        'feature_threshold': float(feature_threshold),
    }
