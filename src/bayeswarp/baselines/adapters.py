from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaselineAdapter(ABC):
    @abstractmethod
    def generate(self, seed_batch, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


class ExternalRepoAdapter(BaselineAdapter):
    """
    Plug in the official implementation of a baseline and emit the same fields used
    by BayesWarp evaluation: failures, time_sec, seed_idx, seed_x, seed_y.
    """
    def __init__(self, name: str):
        self.name = name

    def generate(self, seed_batch, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError(
            f'Connect the official implementation of {self.name} here instead of '            'claiming an unverifiable re-implementation.'
        )
