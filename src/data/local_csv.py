"""Local CSV series — the third data source next to EconData and FRED.

WHY this exists: the SARB MARKET_RATES dataflow has no platinum series and
FRED carries no free daily platinum history, so the team hand-downloaded a
daily USD/troy-ounce CSV. This module normalises such files into the same
data/raw/<name>.parquet shape the API clients produce, so build_dataset.py
treats all three sources identically and stays source-agnostic.

Conversion runs during the `fetch` stage. The source CSVs live in data/raw/
and ARE committed to git (unlike the parquet cache) — they cannot be
re-downloaded by code, so the repo must carry them.
"""
from __future__ import annotations

import pandas as pd

from src.config import DATA_RAW, LOCAL_CSV_SERIES


def convert_local_csvs(force: bool = False) -> dict[str, pd.DataFrame]:
    """Normalise each configured CSV into data/raw/<name>.parquet.

    Expected file shape: two columns, date then value (header names are
    ignored on purpose — export sites name them inconsistently). Dates are
    parsed as US-style MM/DD/YYYY, which we verified against unambiguous
    rows like 07/22/2026 in the platinum file.
    """
    out: dict[str, pd.DataFrame] = {}
    for name, filename in LOCAL_CSV_SERIES.items():
        src = DATA_RAW / filename
        dst = DATA_RAW / f"{name}.parquet"
        if not src.exists():
            raise FileNotFoundError(
                f"Local CSV {src} for series {name!r} is missing — it should "
                "be committed in the repo (see src/data/local_csv.py)."
            )
        if force or not dst.exists():
            # utf-8-sig strips the BOM that spreadsheet exports often prepend.
            raw = pd.read_csv(src, encoding="utf-8-sig")
            tidy = pd.DataFrame(
                {
                    "date": pd.to_datetime(raw.iloc[:, 0], format="%m/%d/%Y"),
                    # coerce: export files sometimes hold blanks or "." for
                    # market holidays — those rows are dropped, matching how
                    # the API clients treat missing observations.
                    "value": pd.to_numeric(raw.iloc[:, 1], errors="coerce"),
                }
            ).dropna().sort_values("date").reset_index(drop=True)
            # Prices of zero are data errors, not prices (the platinum file
            # has one: 2024-10-25 = 0.0), and they poison log-features with
            # -inf. Treat non-positive values as missing — the Friday grid
            # then carries the previous day's price forward, same as any
            # other missing observation.
            tidy = tidy[tidy["value"] > 0].reset_index(drop=True)
            tidy.to_parquet(dst, index=False)
            print(f"cached {name} <- {filename} ({len(tidy)} obs, "
                  f"{tidy['date'].iloc[0].date()}..{tidy['date'].iloc[-1].date()})")
        out[name] = pd.read_parquet(dst)
    return out


if __name__ == "__main__":
    convert_local_csvs(force=True)
