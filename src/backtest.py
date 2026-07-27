"""Expanding-window walk-forward backtest — the ONE place origins loop.

For every Friday origin in the evaluation window:
  1. truncate the weekly table at the origin (this line IS the
     pseudo-real-time guarantee — models physically cannot see past it),
  2. refit every model on that truncated table (expanding window),
  3. forecast log USDZAR h weeks ahead from the origin row,
  4. record (origin, target_date, model, forecast, actual).

Results go to data/results/forecasts.parquet in both log and level units.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    DATA_PROCESSED,
    DATA_RESULTS,
    EVAL_END,
    EVAL_START,
    HORIZON_WEEKS,
    TARGET_COL,
)
from src.models import get_models
from src.models.base import Model


def friday_origins(df: pd.DataFrame) -> pd.DatetimeIndex:
    """All Fridays in the evaluation window that exist in the weekly table."""
    return df.index[(df.index >= EVAL_START) & (df.index <= EVAL_END)]


def run_backtest(
    df: pd.DataFrame,
    models: list[Model],
    origins: pd.DatetimeIndex,
    horizon: int = HORIZON_WEEKS,
) -> pd.DataFrame:
    """Generic engine: any weekly table, any models, any origin list.

    Kept free of file I/O and of model-specific logic so the no-look-ahead
    test in tests/test_backtest.py exercises EXACTLY the code path used in
    production runs.
    """
    rows: list[dict] = []
    for origin in origins:
        # THE pseudo-real-time line: everything a model receives derives
        # from this truncation at the origin Friday.
        train_df = df.loc[df.index <= origin]
        origin_row = df.loc[origin]

        # The grid is regular W-FRI, so origin + h weeks is again a Friday.
        target_date = origin + pd.Timedelta(weeks=horizon)
        # Actual outcome, if that Friday is already in the table. Late-2025
        # origins whose target falls past the data edge keep NaN and are
        # simply excluded from error metrics.
        actual_log = (
            float(df.at[target_date, TARGET_COL]) if target_date in df.index else np.nan
        )

        for model in models:
            model.fit(train_df)  # expanding window: refit at EVERY origin
            forecast_log = model.predict(origin_row)
            rows.append(
                {
                    "origin": origin,
                    "target_date": target_date,
                    "model": model.name,
                    # log units (what models produce) + level units (ZAR per
                    # USD, the unit the task defines RMSE in).
                    "origin_log": float(origin_row[TARGET_COL]),
                    "forecast_log": forecast_log,
                    "actual_log": actual_log,
                    "origin_spot": float(np.exp(origin_row[TARGET_COL])),
                    "forecast": float(np.exp(forecast_log)),
                    "actual": float(np.exp(actual_log)) if not np.isnan(actual_log) else np.nan,
                }
            )
        if origin == origins[0] or origin.month != (origin - pd.Timedelta(weeks=1)).month:
            print(f"  backtest at {origin.date()} (train n={len(train_df)})")
    return pd.DataFrame(rows)


def main() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PROCESSED / "weekly.parquet")
    origins = friday_origins(df)
    print(f"{len(origins)} Friday origins from {origins[0].date()} to {origins[-1].date()}")
    results = run_backtest(df, get_models(), origins)
    path = DATA_RESULTS / "forecasts.parquet"
    results.to_parquet(path, index=False)
    print(f"wrote {path} ({len(results)} forecasts)")
    return results


if __name__ == "__main__":
    main()
