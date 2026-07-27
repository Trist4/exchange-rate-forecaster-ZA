"""Common model interface + the one helper that controls leakage.

Every model implements:
    fit(train_df)              -> learn from all rows up to the origin
    predict(origin_row) -> float  forecast of log USDZAR h weeks after origin

train_df is the weekly table truncated at the origin Friday by the
backtest engine; origin_row is that Friday's row. Models never see
anything else, which is what makes the whole exercise pseudo-real-time.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from src.config import FEATURE_COLS, HORIZON_WEEKS, TARGET_COL


class Model(ABC):
    """Interface shared by every forecaster (the only classes in the repo —
    the brief's model zoo genuinely needs polymorphism)."""

    name: str = "base"

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> None: ...

    @abstractmethod
    def predict(self, origin_row: pd.Series) -> float:
        """Return the forecast of log USDZAR at origin + HORIZON_WEEKS."""


def make_supervised_pairs(
    train_df: pd.DataFrame,
    horizon: int = HORIZON_WEEKS,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build (X_t, y_{t+h}) training pairs WITHOUT leakage.

    Returns (X, y_future, y_now) where X = features at time t,
    y_future = log rate h weeks later, y_now = log rate at t (so models
    can regress the h-week CHANGE, y_future - y_now, on X).

    `features` defaults to the official config list; ModelTester.ipynb
    passes subsets to experiment with alternative feature sets.

    The no-leakage argument: train_df already
    ends at the origin, and shift(-h) makes the last h rows' outcomes NaN,
    so they are dropped. Every surviving pair has BOTH its features and
    its outcome dated on or before the origin — nothing the forecaster
    couldn't have seen that Friday.
    """
    features = features if features is not None else FEATURE_COLS
    y_future = train_df[TARGET_COL].shift(-horizon)
    keep = y_future.notna()
    return (
        train_df.loc[keep, features],
        y_future[keep],
        train_df.loc[keep, TARGET_COL],
    )
