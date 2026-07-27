"""Fundamentals-based regressions: OLS, Ridge, Lasso, SVR, and hybrids.

Common design:

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
from sklearn.svm import SVR

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
    feature selection, so the surviving features are directly readable."""

    name = "Lasso"

    def __init__(self, features: list[str] | None = None) -> None:
        super().__init__(Lasso(random_state=RANDOM_SEED, max_iter=50_000), features)


class LassoResidualRW(Model):
    """Lasso + a random-walk-style correction fitted on Lasso's residuals.

    Two-stage idea ('structural + statistical'):
      1. Lasso forecasts the 4-week change from fundamentals — keeps the
         interpretable sparse-coefficient story.
      2. Whatever Lasso has been systematically missing shows up in its
         recent residuals. The most recent COMPLETED residual is known at
         the origin (its outcome is the origin Friday itself), so we add
         a persistence correction:

             forecast = y_t + lasso_change(X_t) + rho * r_latest

    rho controls how much of the last miss we carry forward. rho=1 is the
    literal 'random walk on the residuals'. By default rho is ESTIMATED in
    the training window (regress residuals on their own 4-week lag) — so
    if misses do not actually persist at the 4-week horizon, the model
    learns rho ~ 0 and safely collapses back to plain Lasso, rather than
    doubling yesterday's error. WHY 4-week lag: consecutive weekly
    residuals overlap by 3 weeks of outcome, so their lag-1 correlation is
    mechanical; lag 4 is the first genuinely independent step.

    Pseudo-real-time check: residuals use only outcomes dated on or before
    the origin — same guarantee as everything else, and the no-look-ahead
    test covers this model automatically once registered.
    """

    name = "Lasso+RW"

    def __init__(self, features: list[str] | None = None, rho: float | None = None) -> None:
        self.features = features if features is not None else FEATURE_COLS
        self._fixed_rho = rho          # None -> estimate from training window
        self._lasso = LassoModel(features)

    def fit(self, train_df: pd.DataFrame) -> None:
        self._lasso.fit(train_df)
        X, y_future, y_now = make_supervised_pairs(train_df, features=self.features)
        # In-sample residuals of the fitted Lasso, one per weekly pair.
        # (Slightly optimistic because Lasso saw these outcomes, but Lasso
        # barely fits anything, so they are close to true forecast errors.)
        resid = (y_future - y_now).to_numpy() - self._lasso._best.predict(X.to_numpy())
        # Residual of the newest completed pair: features at origin-4,
        # outcome at the origin itself -> observable when forecasting.
        self._r_latest = float(resid[-1])
        if self._fixed_rho is not None:
            self._rho = float(self._fixed_rho)
        else:
            r_now, r_lag = resid[4:], resid[:-4]   # 4 weeks apart on the weekly grid
            denom = float(r_lag @ r_lag)
            self._rho = float(r_lag @ r_now / denom) if denom > 0 else 0.0

    def predict(self, origin_row: pd.Series) -> float:
        return float(self._lasso.predict(origin_row) + self._rho * self._r_latest)


class SVMModel(Model):
    """Support-vector regression (RBF kernel) on the 4-week change.

    WHY it's in the zoo: it is our only NONLINEAR model. The linear models
    all conclude the features carry little linear signal; SVR tests the
    remaining hypothesis — that the signal exists but is nonlinear (e.g.
    risk-off only matters when VIX jumps a LOT). Its epsilon-insensitive
    loss also ignores small errors, which acts like Lasso's shrinkage: if
    there is nothing to learn, it can sit near a flat "no change" forecast
    instead of chasing noise.

    Same anti-leakage machinery as Ridge/Lasso: scaling + hyperparameters
    (C = flexibility, epsilon = error tolerance) chosen by time-series CV
    strictly inside each training window. Grid kept small on purpose —
    3x3 combos keep the 261-origin backtest tractable and honest (a huge
    grid tuned on so little validation data would itself be overfitting).
    SVR is deterministic, so no random_state is needed.
    """

    name = "SVM"

    def __init__(self, features: list[str] | None = None) -> None:
        self.features = features if features is not None else FEATURE_COLS
        self._pipe = Pipeline(
            [("scale", StandardScaler()), ("reg", SVR(kernel="rbf", gamma="scale"))]
        )

    def fit(self, train_df: pd.DataFrame) -> None:
        X, y_future, y_now = make_supervised_pairs(train_df, features=self.features)
        search = GridSearchCV(
            self._pipe,
            {
                "reg__C": [0.1, 1.0, 10.0],
                # typical 4-week move is ~2-3% in logs; epsilon spans
                # "ignore almost nothing" to "ignore most weeks".
                "reg__epsilon": [0.001, 0.01, 0.03],
            },
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
