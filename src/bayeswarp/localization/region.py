from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from scipy import ndimage


@dataclass
class Box:
    y1: int
    x1: int
    y2: int
    x2: int
    saliency_mass: float = 0.0

    @property
    def area(self) -> int:
        return max(0, self.y2 - self.y1) * max(0, self.x2 - self.x1)

    def centroid(self) -> Tuple[float, float]:
        return ((self.y1 + self.y2) / 2.0, (self.x1 + self.x2) / 2.0)


def iou(a: Box, b: Box) -> float:
    iy1, ix1 = max(a.y1, b.y1), max(a.x1, b.x1)
    iy2, ix2 = min(a.y2, b.y2), min(a.x2, b.x2)
    inter = max(0, iy2 - iy1) * max(0, ix2 - ix1)
    union = a.area + b.area - inter + 1e-8
    return inter / union


def centroid_distance(a: Box, b: Box) -> float:
    ay, ax = a.centroid()
    by, bx = b.centroid()
    return float(np.sqrt((ay - by) ** 2 + (ax - bx) ** 2))


def merge_boxes(a: Box, b: Box) -> Box:
    return Box(
        y1=min(a.y1, b.y1),
        x1=min(a.x1, b.x1),
        y2=max(a.y2, b.y2),
        x2=max(a.x2, b.x2),
        saliency_mass=a.saliency_mass + b.saliency_mass,
    )


def top_alpha_mask(saliency: np.ndarray, alpha: float) -> np.ndarray:
    threshold = np.quantile(saliency, 1 - alpha)
    return (saliency > threshold).astype(np.uint8)


def extract_components(mask: np.ndarray, saliency: np.ndarray, area_min: int) -> List[Box]:
    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, num = ndimage.label(mask, structure=structure)
    boxes = []
    for lab in range(1, num + 1):
        ys, xs = np.where(labeled == lab)
        if len(ys) < area_min:
            continue
        y1, x1, y2, x2 = ys.min(), xs.min(), ys.max() + 1, xs.max() + 1
        mass = float(saliency[y1:y2, x1:x2].sum())
        boxes.append(Box(y1, x1, y2, x2, mass))
    return boxes


def merge_nearby_boxes(boxes: List[Box], tau_iou: float, d_max: float) -> List[Box]:
    changed = True
    boxes = boxes[:]
    while changed and len(boxes) > 1:
        changed = False
        used = [False] * len(boxes)
        merged = []
        for i in range(len(boxes)):
            if used[i]:
                continue
            curr = boxes[i]
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                if iou(curr, boxes[j]) > tau_iou or centroid_distance(curr, boxes[j]) < d_max:
                    curr = merge_boxes(curr, boxes[j])
                    used[j] = True
                    changed = True
            used[i] = True
            merged.append(curr)
        boxes = merged
    return boxes


def select_boxes_by_budget(boxes: List[Box], image_shape: Tuple[int, int], rho: float) -> List[Box]:
    H, W = image_shape
    max_area = rho * H * W
    selected = []
    used_area = 0
    for box in sorted(boxes, key=lambda b: b.saliency_mass, reverse=True):
        if used_area + box.area <= max_area or not selected:
            selected.append(box)
            used_area += box.area
    return selected


def boxes_to_mask(boxes: List[Box], image_shape: Tuple[int, int]) -> np.ndarray:
    H, W = image_shape
    mask = np.zeros((H, W), dtype=np.float32)
    for b in boxes:
        mask[b.y1:b.y2, b.x1:b.x2] = 1.0
    return mask


def localize_critical_region(saliency: np.ndarray, alpha: float, area_min: int, tau_iou: float, d_max: float, rho: float):
    mask = top_alpha_mask(saliency, alpha)
    boxes = extract_components(mask, saliency, area_min)
    boxes = merge_nearby_boxes(boxes, tau_iou=tau_iou, d_max=d_max)
    boxes = select_boxes_by_budget(boxes, saliency.shape, rho)
    if len(boxes) == 0:
        fallback_mask = top_alpha_mask(saliency, alpha).astype(np.float32)
        if fallback_mask.sum() == 0:
            fallback_mask = np.ones_like(saliency, dtype=np.float32)
        return fallback_mask, []
    return boxes_to_mask(boxes, saliency.shape), boxes
