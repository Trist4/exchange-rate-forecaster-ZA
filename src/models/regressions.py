"""Fundamentals-based regressions: OLS, Ridge, Lasso.

Common design (worth being able to recite to a judge):

* We regress the 4-week log CHANGE (y_{t+h} - y_t) on the features, not
  the level. FX levels are near unit-root; regressing levels on slowly
  moving features produces spurious fits that fall apart out of sample.
  The final forecast is origin log level + predicted change.

* Ridge/Lasso hyperparameters are tuned with TimeSeriesSplit INSIDE each
  training window only. TimeSeriesSplit always validates on a block that
  comes chronologically AFTER its training folds, and the whole search
  sees nothing past the origin — so tuning cannot leak future data.

* If a feature is missing at the origin (only possible at sample edges),
  we fall back to the no-change forecast instead of crashing — a real
  desk would do the same rather than extrapolate through a data gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_COLS, RANDOM_SEED, TARGET_COL
from src.models.base import Model, make_supervised_pairs

# One shared alpha grid: wide in log-space because the right penalty for
# weekly FX data is unknown a priori and varies with sample size.
ALPHA_GRID = np.logspace(-4, 2, 25)


def _origin_features(origin_row: pd.Series, features: list[str]) -> np.ndarray | None:
    x = origin_row[features].to_numpy(dtype=float)
    return None if np.isnan(x).any() else x.reshape(1, -1)


class FundamentalsOLS(Model):
    """Plain OLS of the 4-week change on all features.

    WHY keep it despite Ridge/Lasso existing: it is the fully transparent
    version — every coefficient has a sign we can defend economically
    (e.g. gold up => rand stronger => negative coefficient).

    `features=None` -> the official config list; pass a subset in
    ModelTester.ipynb to experiment with alternative feature sets.
    """

    name = "OLS"

    def __init__(self, features: list[str] | None = None) -> None:
        self.features = features if features is not None else FEATURE_COLS

    def fit(self, train_df: pd.DataFrame) -> None:
        X, y_future, y_now = make_supervised_pairs(train_df, features=self.features)
        self._fit = sm.OLS(y_future - y_now, sm.add_constant(X)).fit()

    def predict(self, origin_row: pd.Series) -> float:
        x = _origin_features(origin_row, self.features)
        if x is None:
            return float(origin_row[TARGET_COL])  # fallback: no-change
        exog = np.concatenate(([1.0], x.ravel()))
        return float(origin_row[TARGET_COL] + self._fit.params @ exog)


class _TunedLinear(Model):
    """Shared machinery for Ridge/Lasso: scale -> penalised regression,
    alpha chosen by time-series CV within the training window."""

    def __init__(self, estimator, features: list[str] | None = None) -> None:
        self.features = features if features is not None else FEATURE_COLS
        # StandardScaler first: penalties shrink all coefficients by the
        # same alpha, which is only fair if features share a scale.
        self._pipe = Pipeline([("scale", StandardScaler()), ("reg", estimator)])

    def fit(self, train_df: pd.DataFrame) -> None:
        X, y_future, y_now = make_supervised_pairs(train_df, features=self.features)
        search = GridSearchCV(
            self._pipe,
            {"reg__alpha": ALPHA_GRID},
            cv=TimeSeriesSplit(n_splits=4),
            scoring="neg_mean_squared_error",
        )
        search.fit(X.to_numpy(), (y_future - y_now).to_numpy())
        self._best = search.best_estimator_

    def predict(self, origin_row: pd.Series) -> float:
        x = _origin_features(origin_row, self.features)
        if x is None:
            return float(origin_row[TARGET_COL])  # fallback: no-change
        return float(origin_row[TARGET_COL] + self._best.predict(x)[0])


class RidgeModel(_TunedLinear):
    """Ridge: keeps every feature but shrinks noisy coefficients toward 0.
    Sensible when predictors are correlated (gold/platinum/dollar all are)."""

    name = "Ridge"

    def __init__(self, features: list[str] | None = None) -> None:
        super().__init__(Ridge(random_state=RANDOM_SEED), features)


class LassoModel(_TunedLinear):
    """Lasso: shrinks AND zeroes out features — doubles as automatic
    feature selection, and the surviving features make a great demo slide."""

    name = "Lasso"

    def __init__(self, features: list[str] | None = None) -> None:
        super().__init__(Lasso(random_state=RANDOM_SEED, max_iter=50_000), features)
