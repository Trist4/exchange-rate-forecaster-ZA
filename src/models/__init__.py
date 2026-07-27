"""Model registry: one place the backtest asks for 'all models'."""
from __future__ import annotations

from src.models.base import Model
from src.models.benchmarks import AR1Level, RandomWalk, RandomWalkDrift
from src.models.regressions import (
    FundamentalsOLS,
    LassoModel,
    LassoResidualRW,
    RidgeModel,
    SVMModel,
)

# Name of the official benchmark — evaluate.py compares everything to it.
BENCHMARK_NAME = AR1Level.name


def get_models() -> list[Model]:
    """Fresh instances every call so backtests never share fitted state."""
    return [
        RandomWalk(),
        RandomWalkDrift(),
        AR1Level(),
        FundamentalsOLS(),
        RidgeModel(),
        LassoModel(),
        SVMModel(),
        LassoResidualRW(),
    ]
