"""Baseline DNN testing methods, run under BayesWarp's controlled conditions."""
from __future__ import annotations

from typing import Dict, Type

from baselines.common import Baseline, BudgetedOracle, BudgetExhausted
from baselines.adapt import Adapt
from baselines.nsgen import NSGen
from baselines.suntest import SUNTest

BASELINES: Dict[str, Type[Baseline]] = {
    'adapt': Adapt,
    'nsgen': NSGen,
    'suntest': SUNTest,
}

__all__ = ['Baseline', 'BudgetedOracle', 'BudgetExhausted', 'BASELINES', 'Adapt', 'NSGen', 'SUNTest']
