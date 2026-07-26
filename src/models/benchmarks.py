"""Benchmark models: random walk, random walk with drift, AR(1).

These are deliberately tiny — the exchange-rate literature (Meese-Rogoff
onwards) says they are brutally hard to beat at short horizons, which is
exactly why the brief scores everything against AR(1).
"""
from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

from src.config import HORIZON_WEEKS, TARGET_COL
from src.models.base import Model


class RandomWalk(Model):
    """Forecast = today's rate ('no-change'). The classic FX benchmark."""

    name = "RandomWalk"

    def fit(self, train_df: pd.DataFrame) -> None:
        pass  # nothing to estimate — that is the point of this benchmark

    def predict(self, origin_row: pd.Series) -> float:
        return float(origin_row[TARGET_COL])


class RandomWalkDrift(Model):
    """Forecast = today's rate + h * average weekly drift.

    WHY: USDZAR has trended up for decades (SA-US inflation differential),
    so 'no change plus the long-run trend' is the natural second benchmark.
    The drift is re-estimated from the training window at every origin.
    """

    name = "RW+Drift"

    def fit(self, train_df: pd.DataFrame) -> None:
        self._drift = float(train_df[TARGET_COL].diff().mean())

    def predict(self, origin_row: pd.Series) -> float:
        return float(origin_row[TARGET_COL]) + HORIZON_WEEKS * self._drift


class AR1Level(Model):
    """OFFICIAL BENCHMARK: AR(1) on the log level.

    Implemented as a DIRECT h-step projection: OLS of y_{t+h} on y_t.
    WHY direct rather than iterating a 1-week AR(1) four times: one
    regression, no compounding of one-step parameter error, and the
    fitted slope is directly interpretable ('how much of today's level
    survives 4 weeks' — empirically close to 1, i.e. near-random-walk).
    """

    name = "AR1"

    def fit(self, train_df: pd.DataFrame) -> None:
        y = train_df[TARGET_COL]
        y_future = y.shift(-HORIZON_WEEKS)
        keep = y_future.notna()
        X = sm.add_constant(y[keep])          # [1, y_t]
        self._params = sm.OLS(y_future[keep], X).fit().params.to_numpy()

    def predict(self, origin_row: pd.Series) -> float:
        const, slope = self._params
        return float(const + slope * origin_row[TARGET_COL])
