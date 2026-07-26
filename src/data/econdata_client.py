"""EconData (SARB) access with a discovery-first workflow.

We do NOT hard-code dataset IDs or series keys anywhere in this repo:
the exact identifiers are discovered interactively (see `discover` below /
the CLI at the bottom) and then pasted into src/config.py. This keeps the
pipeline honest — nothing is silently guessed.

Fetch-once cache: every configured series is stored as
data/raw/<name>.parquet (columns: date, value). All downstream code reads
only these parquet files, so the pipeline runs offline after one fetch and
every run uses byte-identical inputs (reproducibility for the judges).
"""
from __future__ import annotations

import os
import sys

import pandas as pd
from dotenv import load_dotenv

from src.config import DATA_RAW

# Make ECONDATA_CREDENTIALS from the git-ignored .env visible to econdatapy.
load_dotenv()


def _read_dataset(dataset_id: str) -> dict:
    """Call econdatapy for one dataset, with a friendly error if unconfigured."""
    if not dataset_id:
        raise RuntimeError(
            "ECONDATA_DATASET_ID is empty in src/config.py. Run the discovery "
            "helper first:  python -m src.data.econdata_client <DATASET_ID>"
        )
    if not os.getenv("ECONDATA_CREDENTIALS"):
        raise RuntimeError(
            "ECONDATA_CREDENTIALS is not set. Copy .env.example to .env and "
            "paste your EconData credentials."
        )
    # Imported lazily so the rest of the pipeline (models, backtest, tests)
    # works even before econdatapy is installed from test PyPI.
    from econdatapy import read

    return read.dataset(dataset_id)


def _series_label(frame: pd.DataFrame) -> str:
    """Best-effort human label for a series.

    econdatapy attaches the per-series SDMX attributes as an ad-hoc
    `frame.metadata` dict (verified by reading its process_series source);
    we probe the usual descriptive keys and otherwise dump the dict
    compactly so discovery is still informative.
    """
    meta = getattr(frame, "metadata", None)
    if not isinstance(meta, dict):
        meta = getattr(frame, "attrs", None)  # future-proofing fallback
    if isinstance(meta, dict) and meta:
        for k in ("LABEL", "label", "NAME", "name", "TITLE", "title",
                  "DESCRIPTION", "description"):
            if k in meta:
                return str(meta[k])
        return "; ".join(
            f"{k}={v}" for k, v in meta.items() if k != "series-key"
        )[:120]
    return ""


def discover(dataset_id: str, filter_text: str | None = None) -> None:
    """Print every series key (+ label if available) in a dataset.

    Usage: python -m src.data.econdata_client MRD          # list everything
           python -m src.data.econdata_client MRD EXCX135  # filter rows
    Use this to locate EXCX135 (USDZAR), gold, platinum, JIBAR/T-bill and
    the 10y bond yield, then paste the exact keys into src/config.py.
    """
    result = _read_dataset(dataset_id)
    data = result.get("data", {})
    print(f"Dataset {dataset_id!r}: {len(data)} series")
    print("-" * 78)
    shown = 0
    for key, frame in data.items():
        line = f"{key:<30} {_series_label(frame)}"
        if filter_text and filter_text.lower() not in line.lower():
            continue
        print(line)
        shown += 1
    print("-" * 78)
    print(f"{shown} series shown" + (f" (filter: {filter_text!r})" if filter_text else ""))
    # The dataset-level metadata often carries dimension code lists that
    # help decode the keys — print its top-level shape as a breadcrumb.
    meta = result.get("metadata")
    if meta is not None:
        print(f"\nmetadata type: {type(meta).__name__}; "
              f"keys: {list(meta)[:10] if hasattr(meta, '__iter__') else meta}")


def _tidy(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise an econdatapy series frame to columns (date, value)."""
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["TIME_PERIOD"]),
            # OBS_VALUE sometimes arrives as strings; coerce and drop blanks.
            "value": pd.to_numeric(frame["OBS_VALUE"], errors="coerce"),
        }
    )
    return out.dropna().sort_values("date").reset_index(drop=True)


def fetch_sarb_series(
    dataset_id: str, series_map: dict[str, str], force: bool = False
) -> dict[str, pd.DataFrame]:
    """Fetch the configured SARB series, hitting the API at most once.

    series_map maps our short name -> exact series key from discovery.
    Returns {name: DataFrame(date, value)}; each series is cached at
    data/raw/<name>.parquet and the API is only called for cache misses.
    """
    unset = [name for name, key in series_map.items() if not key]
    if unset:
        raise RuntimeError(
            f"SARB_SERIES entries not filled in src/config.py: {unset}. "
            "Run the discovery helper and paste the exact series keys."
        )

    missing = {
        name: key
        for name, key in series_map.items()
        if force or not (DATA_RAW / f"{name}.parquet").exists()
    }
    if missing:
        data = _read_dataset(dataset_id)["data"]
        for name, key in missing.items():
            # Exact match first; otherwise accept a unique key that CONTAINS
            # the configured code (keys often carry extra dimensions like
            # frequency, e.g. "EXCX135.B"). Ambiguity is an error, not a guess.
            if key in data:
                matched = key
            else:
                candidates = [k for k in data if key in k]
                if len(candidates) != 1:
                    raise KeyError(
                        f"Series {key!r} for {name!r}: found {candidates or 'nothing'} "
                        f"in dataset {dataset_id!r}. Re-run discovery and use the exact key."
                    )
                matched = candidates[0]
            _tidy(data[matched]).to_parquet(DATA_RAW / f"{name}.parquet", index=False)
            print(f"cached {name} <- {matched}")

    return {
        name: pd.read_parquet(DATA_RAW / f"{name}.parquet") for name in series_map
    }


if __name__ == "__main__":
    # Discovery CLI: python -m src.data.econdata_client <DATASET_ID> [filter]
    if len(sys.argv) < 2:
        print(__doc__)
        print("usage: python -m src.data.econdata_client <DATASET_ID> [filter-text]")
        sys.exit(1)
    discover(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
