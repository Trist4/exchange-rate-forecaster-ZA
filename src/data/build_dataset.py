"""Build the tidy Friday-sampled weekly modelling table.

Reads ONLY the parquet cache in data/raw/ (never the network), so the
modelling pipeline is reproducible and runs offline. Output:
data/processed/weekly.parquet — one row per Friday, columns = raw levels
+ target + features, indexed by the Friday date.

Pseudo-real-time rule enforced here: every value on the row for Friday t
is a function of observations dated ON OR BEFORE t only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    DATA_PROCESSED,
    DATA_RAW,
    DATA_START,
    FEATURE_COLS,
    FRED_SERIES,
    LOCAL_CSV_SERIES,
    SARB_SERIES,
    TARGET_COL,
)

# Every series the table needs, from any source. If a cache file is
# missing we fail loudly with instructions instead of fetching silently.
ALL_SERIES = list(SARB_SERIES) + list(FRED_SERIES) + list(LOCAL_CSV_SERIES)


def load_cached_series() -> dict[str, pd.Series]:
    """Load each cached series as a date-indexed Series of floats."""
    out: dict[str, pd.Series] = {}
    missing = []
    for name in ALL_SERIES:
        path = DATA_RAW / f"{name}.parquet"
        if not path.exists():
            missing.append(name)
            continue
        frame = pd.read_parquet(path)
        out[name] = frame.set_index("date")["value"].sort_index()
    if missing:
        raise FileNotFoundError(
            f"Missing cached series {missing} in data/raw/. "
            "Run `python run.py fetch` first (SARB series also need the "
            "discovery step — see src/data/econdata_client.py)."
        )
    return out


def to_friday_grid(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Sample every daily series onto a common Friday grid.

    For each Friday we take the LAST observation on or before that Friday
    (pandas .asof). WHY: markets close for holidays (Good Friday, SA/US
    public holidays), so some Fridays simply have no print — carrying the
    most recent value forward is what a forecaster standing on that Friday
    would actually see. Using the next Monday instead would be look-ahead.
    """
    last_common = min(s.index.max() for s in series.values())
    fridays = pd.date_range(DATA_START, last_common, freq="W-FRI")
    df = pd.DataFrame(index=fridays)
    df.index.name = "friday"
    for name, s in series.items():
        df[name] = s.asof(fridays).to_numpy()
    return df


def add_target_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the log target and the predictor columns (see config.FEATURE_COLS).

    Everything below uses only same-row-or-earlier data (diffs and rolling
    means look backwards), so each row is fully observable at its own
    Friday. The forward-looking pairing X_t -> y_{t+h} happens inside the
    models' fit(), never here.
    """
    out = df.copy()

    # Target in logs: log changes are symmetric % moves and the random-walk
    # benchmark is exact in logs.
    out[TARGET_COL] = np.log(out["usdzar"])

    # 4-week log changes of commodity/dollar prices: matches the 4-week
    # forecast horizon, and momentum in SA's terms-of-trade drivers is the
    # economic story we want the models to exploit. (Platinum comes from a
    # hand-downloaded CSV — see LOCAL_CSV_SERIES in config.py.)
    for name in ("gold", "platinum", "brent", "dxy"):
        out[f"d4_log_{name}"] = np.log(out[name]).diff(4)

    # VIX is already a percentage-like index, so a simple 4-week difference
    # (not a log change) is the natural "risk appetite shifted" measure.
    out["d4_vix"] = out["vix"].diff(4)

    # US credit spread (Moody's Baa yield minus 10y Treasury), in
    # percentage points, so same treatment: 4-week difference = "did US
    # credit stress rise or fall". WHY it belongs: credit spreads are the
    # classic risk-off gauge for EM FX and move on stress that VIX (an
    # equity measure) sometimes misses. (See config.py for why this stands
    # in for the ICE BofA HY spread, which FRED truncated to ~3 years.)
    out["d4_credit"] = (out["baa"] - out["us_10y"]).diff(4)

    # Carry / interest-parity term: SA short rate minus US short rate, in
    # levels because the spread itself (not its change) prices the carry.
    out["rate_spread"] = out["sa_3m"] - out["us_3m"]

    # Mean-reversion term: how far (in log points) the rand sits from its
    # own trailing 52-week average. min_periods=52 so early rows are NaN
    # rather than being computed from a short, unrepresentative window.
    out["mr_gap"] = out[TARGET_COL] - out[TARGET_COL].rolling(52, min_periods=52).mean()

    # Keep only rows where target + all features exist (the first ~52 weeks
    # fall away because of the rolling mean — that is expected and cheap
    # given DATA_START is ~11 years before the first forecast origin).
    return out.dropna(subset=[TARGET_COL, *FEATURE_COLS])


def main() -> pd.DataFrame:
    series = load_cached_series()
    weekly = add_target_and_features(to_friday_grid(series))
    path = DATA_PROCESSED / "weekly.parquet"
    weekly.to_parquet(path)
    print(f"wrote {path} — {len(weekly)} Fridays, {weekly.index.min().date()} "
          f"to {weekly.index.max().date()}")
    return weekly


if __name__ == "__main__":
    main()
