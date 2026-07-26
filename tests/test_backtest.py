"""Proof of pseudo-real-time discipline, on synthetic data.

The key test: corrupt every row AFTER an origin Friday and check the
forecasts made AT that origin do not move by a single bit. If any model
or the engine ever peeks past the origin, this fails loudly.

Synthetic data (seeded, deterministic) so the test runs before any real
data is downloaded and in CI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import run_backtest
from src.config import FEATURE_COLS, HORIZON_WEEKS, RANDOM_SEED, TARGET_COL
from src.models import get_models


def make_synthetic_weekly(n: int = 220) -> pd.DataFrame:
    """A weekly table with the real schema: random-walk log target plus
    noise features that weakly predict it (so Lasso keeps some alive)."""
    rng = np.random.default_rng(RANDOM_SEED)
    idx = pd.date_range("2018-01-05", periods=n, freq="W-FRI")
    df = pd.DataFrame(index=idx)
    df.index.name = "friday"
    shocks = rng.normal(0.0, 0.02, n)
    df[TARGET_COL] = np.log(15.0) + shocks.cumsum()
    for i, col in enumerate(FEATURE_COLS):
        # Feature i mildly correlates with future shocks to make the
        # regressions non-degenerate; exact form is irrelevant to the test.
        df[col] = rng.normal(0.0, 1.0, n) + 0.3 * np.roll(shocks, -1 - i)
    return df


def test_horizon_alignment() -> None:
    """target_date must sit exactly HORIZON_WEEKS Fridays after the origin."""
    df = make_synthetic_weekly()
    origins = df.index[[150]]
    res = run_backtest(df, get_models(), origins)
    assert (res["target_date"] - res["origin"] == pd.Timedelta(weeks=HORIZON_WEEKS)).all()
    # And the recorded actual is genuinely the table value at target_date.
    expected = df.at[origins[0] + pd.Timedelta(weeks=HORIZON_WEEKS), TARGET_COL]
    assert np.allclose(res["actual_log"], expected)


@pytest.mark.parametrize("origin_pos", [120, 160, 200])
def test_no_lookahead(origin_pos: int) -> None:
    """Perturbing data AFTER the origin must not change that origin's
    forecasts — for every model, at several origins."""
    df = make_synthetic_weekly()
    origin = df.index[origin_pos]
    models = get_models()

    baseline = run_backtest(df, models, df.index[[origin_pos]])

    # Violent corruption of the entire post-origin future: if anything
    # downstream of the origin leaks into fit() or predict(), forecasts
    # cannot possibly stay identical.
    corrupted = df.copy()
    future = corrupted.index > origin
    rng = np.random.default_rng(0)
    corrupted.loc[future, :] = rng.normal(100.0, 50.0, size=(future.sum(), df.shape[1]))

    rerun = run_backtest(corrupted, get_models(), corrupted.index[[origin_pos]])

    for model_name in baseline["model"]:
        f0 = baseline.loc[baseline["model"] == model_name, "forecast_log"].iloc[0]
        f1 = rerun.loc[rerun["model"] == model_name, "forecast_log"].iloc[0]
        assert f0 == f1, (
            f"{model_name} forecast at {origin.date()} changed when future "
            f"data was perturbed: {f0} -> {f1}. LOOK-AHEAD BUG."
        )
    # Sanity check that the corruption actually reached the actuals —
    # otherwise this test could pass vacuously.
    assert not np.allclose(baseline["actual_log"], rerun["actual_log"])
