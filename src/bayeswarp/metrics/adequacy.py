from __future__ import annotations
from typing import List, Sequence

import torch


def _nearest(query: torch.Tensor, pool: torch.Tensor, chunk: int = 4096):
    best_distance = None
    best_index = None
    for start in range(0, pool.size(0), chunk):
        block = pool[start:start + chunk]
        distances = torch.cdist(query.unsqueeze(0), block).squeeze(0)
        value, index = distances.min(dim=0)
        if best_distance is None or float(value) < float(best_distance):
            best_distance = value
            best_index = start + int(index.item())
    return best_index, best_distance


def distance_based_surprise_adequacy(
    generated_features: torch.Tensor,
    generated_predictions: Sequence[int],
    reference_features: torch.Tensor,
    reference_predictions: Sequence[int],
) -> List[float]:
    reference_predictions = torch.as_tensor(list(reference_predictions))
    scores: List[float] = []

    for i in range(generated_features.size(0)):
        target = int(generated_predictions[i])
        same_mask = reference_predictions == target
        diff_mask = ~same_mask
        if int(same_mask.sum()) == 0 or int(diff_mask.sum()) == 0:
            scores.append(float('nan'))
            continue

        same_pool = reference_features[same_mask]
        diff_pool = reference_features[diff_mask]

        a_index, dist_a = _nearest(generated_features[i], same_pool)
        x_a = same_pool[a_index]
        _, dist_b = _nearest(x_a, diff_pool)

        denominator = float(dist_b)
        scores.append(float(dist_a) / denominator if denominator > 0 else float('nan'))

    return scores
