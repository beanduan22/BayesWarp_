from __future__ import annotations
from typing import List

import torch
import torch.nn as nn

from bayeswarp.models.factory import find_classifier_head


class PenultimateExtractor:
    def __init__(self, model: nn.Module):
        self.model = model
        self.head = find_classifier_head(model)
        self._buffer = {}
        self._handle = self.head.register_forward_pre_hook(self._hook)

    def _hook(self, _module, inputs):
        self._buffer['value'] = inputs[0].detach()

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        _ = self.model(x)
        return self._buffer['value'].flatten(start_dim=1)

    @torch.no_grad()
    def batched(self, images: List[torch.Tensor], device: torch.device, batch_size: int = 64) -> torch.Tensor:
        features = []
        for start in range(0, len(images), batch_size):
            chunk = images[start:start + batch_size]
            batch = torch.cat([c if c.ndim == 4 else c.unsqueeze(0) for c in chunk], dim=0).to(device)
            features.append(self(batch).cpu())
        if not features:
            return torch.empty(0)
        return torch.cat(features, dim=0)

    def close(self) -> None:
        self._handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
