"""Central configuration for the USDZAR forecasting project.

Every tunable choice lives in this one file so that when a judge asks
"where is that decided?", any team member can point here. Nothing below
is buried inside a function.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"                # cached downloads, one parquet per series
DATA_PROCESSED = ROOT / "data" / "processed"    # the tidy weekly modelling table
DATA_RESULTS = ROOT / "data" / "results"        # walk-forward forecasts
FIGURES = ROOT / "reports" / "figures"          # presentation-quality plots
REPORTS = ROOT / "reports"                      # metric tables (csv)

# ---------------------------------------------------------------------------
# Forecasting task (from the official brief)
# ---------------------------------------------------------------------------
# WHY h = 4 weeks: the brief says "1 month ahead" with a forecast every
# Friday. On a Friday-sampled weekly grid the cleanest, unambiguous reading
# is "4 Fridays ahead" — calendar months have 4 or 5 Fridays, so a literal
# calendar month would make the horizon vary between origins and the
# evaluation would mix horizons. One constant, easy to change if the
# organisers clarify otherwise.
HORIZON_WEEKS: int = 4

# Forecast ORIGINS span this window (the brief's evaluation period).
EVAL_START = "2021-01-01"
EVAL_END = "2025-12-31"

# WHY start in 2010: the expanding window needs a healthy pre-2021 training
# sample (~570 weekly obs by the first origin) and all our series exist by
# then. Earlier data is cheap to add but 2008/09 crisis dynamics are of
# questionable relevance to a 2021-25 evaluation.
DATA_START = "2010-01-01"

# One seed used everywhere randomness appears (sklearn Lasso, synthetic
# test data) so every run of the pipeline is exactly reproducible.
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# EconData / SARB series (filled in AFTER running the discovery helper)
# ---------------------------------------------------------------------------
# We deliberately do NOT guess dataset IDs or series keys. Run:
#
#   python -m src.data.econdata_client <DATASET_ID> [filter-text]
#
# to list every series key + label in a dataset, then paste the exact keys
# below. The USDZAR series is EconData code EXCX135 (SARB -> Rates ->
# Market Rates dataflow); its full series KEY inside the dataset may carry
# extra dimensions (e.g. frequency), so confirm it via discovery too.
# Filled from discovery on 2026-07-26 (see DEVELOPERSETUP.MD §4):
# MARKET_RATES = the SARB "Market Rates" dataflow, 28 daily series.
ECONDATA_DATASET_ID: str = "MARKET_RATES"

SARB_SERIES: dict[str, str] = {
    # our_name : exact series key inside ECONDATA_DATASET_ID
    "usdzar": "EXCX135.B.A",  # Rand per US Dollar
    # Gold in US DOLLARS (GDPL201), not Rand (GDPL203): the USD price is the
    # world commodity price; the Rand price already contains USDZAR itself
    # and would leak the target into a "fundamental".
    "gold": "GDPL201.B.A",
    # NOTE: no platinum series exists in MARKET_RATES (checked via
    # discovery 2026-07-26), so the brief's "platinum if present" clause
    # applies and the d4_log_platinum feature was dropped from FEATURE_COLS.
    # 3m JIBAR: the standard SA money-market reference rate, daily prints.
    # (Alternative: MMSD402.D.A, 91-day T-bill tender rate.)
    "sa_3m": "MMSD502.D.A",
    "sa_10y": "CMJD004.B.A",  # 10 years and longer, daily average bond yields
}

# ---------------------------------------------------------------------------
# FRED series (public, no key needed, codes are stable and documented)
# ---------------------------------------------------------------------------
FRED_SERIES: dict[str, str] = {
    "dxy": "DTWEXBGS",       # broad trade-weighted dollar index
    "vix": "VIXCLS",         # CBOE VIX (global risk appetite)
    "us_3m": "DGS3MO",       # US 3-month Treasury yield
    "brent": "DCOILBRENTEU", # Brent crude (SA is a large oil importer)
}

# ---------------------------------------------------------------------------
# Modelling table schema
# ---------------------------------------------------------------------------
# The target is the LOG exchange rate: FX rates are strictly positive and
# roughly log-normal, log changes are symmetric percentage moves, and the
# random-walk benchmark is exact in logs.
TARGET_COL = "log_usdzar"

# Features available at each Friday origin t (built in build_dataset.py).
# All are functions of data observable ON OR BEFORE t; the h-step lag to
# the outcome is created when models pair X_t with y_{t+h} during fit.
FEATURE_COLS: list[str] = [
    "d4_log_gold",      # 4-week log change in gold (SA's top export)
    # (d4_log_platinum dropped — no platinum series in MARKET_RATES)
    "d4_log_brent",     # 4-week log change in Brent (SA's top import)
    "d4_log_dxy",       # 4-week log change in broad USD index (global USD strength)
    "d4_vix",           # 4-week change in VIX level (risk-on/risk-off for EM FX)
    "rate_spread",      # SA 3m rate minus US 3m rate (carry / UIP-style term)
    "mr_gap",           # log USDZAR minus its trailing 52-week mean (mean reversion)
]
