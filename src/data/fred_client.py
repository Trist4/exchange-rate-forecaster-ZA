"""US-side predictors from FRED via pandas-datareader (no API key needed).

WHY these series: they are daily FINANCIAL prices/yields, observable in
real time with no publication lag and no revisions — so using their value
on Friday t in a forecast made on Friday t is genuinely pseudo-real-time.
(Macro releases like CPI would need a release-calendar to be honest.)

Same fetch-once cache pattern as the SARB client: one parquet per series
in data/raw/, downstream code reads only the cache.
"""
from __future__ import annotations

import pandas as pd
from pandas_datareader import data as pdr

from src.config import DATA_RAW, DATA_START, FRED_SERIES


def fetch_fred_series(force: bool = False) -> dict[str, pd.DataFrame]:
    """Download every configured FRED series (cache misses only).

    Returns {name: DataFrame(date, value)} matching the SARB client shape,
    so build_dataset.py can treat both sources identically.
    """
    out: dict[str, pd.DataFrame] = {}
    for name, code in FRED_SERIES.items():
        path = DATA_RAW / f"{name}.parquet"
        if force or not path.exists():
            raw = pdr.DataReader(code, "fred", start=DATA_START)
            tidy = (
                raw.rename(columns={code: "value"})
                .reset_index()
                .rename(columns={"DATE": "date"})
                .dropna()
            )
            tidy.to_parquet(path, index=False)
            print(f"cached {name} <- FRED:{code} ({len(tidy)} obs)")
        out[name] = pd.read_parquet(path)
    return out


if __name__ == "__main__":
    fetch_fred_series()
