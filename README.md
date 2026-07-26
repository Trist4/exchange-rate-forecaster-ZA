# exchange-rate-forecaster-ZA

Forecasting **USDZAR one month ahead**, with a fresh out-of-sample forecast
made **every Friday** from 2021-01-01 to 2025-12-31, benchmarked against
AR(1). Built for the SES × Codera hackathon.

> **New teammate?** Read [DEVELOPERSETUP.MD](DEVELOPERSETUP.MD) — full
> from-scratch setup, the discovery walkthrough, and a tour of the codebase.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
make install                       # requirements + econdatapy from test PyPI
# macOS/Homebrew only: econdatapy imports tkinter, which Homebrew Python
# doesn't bundle — `brew install python-tk@3.12` (match your Python version)

cp .env.example .env               # then paste your ECONDATA_CREDENTIALS

# One-time discovery: locate the SARB Market Rates dataset & series keys
make discover ID=<DATASET_ID> FILTER=EXCX135
#   ...paste ECONDATA_DATASET_ID and SARB_SERIES keys into src/config.py

make all                           # fetch -> build -> backtest -> evaluate -> plots
make test                          # incl. the no-look-ahead proof
```

Outputs land in `reports/` (metric tables), `reports/figures/` (presentation
plots) and `data/results/forecasts.parquet` (every forecast ever made).
`notebooks/01_exploration.ipynb` renders everything inline for the demo.

## Methodology (the 60-second version)

**Friday grid.** All daily series are sampled to a weekly Friday grid using
the *last observation on or before* each Friday — holidays mean some Fridays
have no print, and grabbing the next available day instead would be
look-ahead.

**"1 month" = 4 weeks.** On a Friday grid the unambiguous reading of "1
month ahead" is `HORIZON_WEEKS = 4` (a calendar month has 4 *or* 5 Fridays,
which would mix horizons across origins). It is a single constant in
`src/config.py` if the organisers prefer another interpretation.

**Pseudo-real-time discipline.** At each Friday origin the engine truncates
the dataset at the origin and refits *every* model on that expanding window;
a forecast can only use data observable on or before its own Friday.
`tests/test_backtest.py` proves it: corrupting all post-origin data changes
no forecast by a single bit.

**Why financial predictors, not macro releases.** CPI/GDP-type series are
published with lags and revised afterwards — using their final vintages
would quietly cheat. Daily market prices (gold, platinum, Brent, the broad
dollar index, VIX, short rates) are observable the moment they print and
never revised, so the backtest is honestly real-time. They also carry the
economic story: SA's terms of trade, global dollar strength, risk appetite
and carry.

**Models.** RandomWalk, RW+drift, **AR(1) (official benchmark)**,
fundamentals OLS, Ridge and Lasso — the latter two tuned with time-series
CV *inside* each training window. All models forecast the log rate;
evaluation is on levels (ZAR per USD).

**Evaluation.** Overall and per-year RMSE/MAE/directional hit-rate, plus a
Diebold–Mariano test (squared-error loss, Harvey small-sample correction)
of every model against AR(1).

## Results

*(placeholder — populated by `make evaluate`, see `reports/metrics_overall.csv`)*

| Model      | RMSE | MAE | Hit rate | RMSE vs AR(1) | DM p-value |
|------------|------|-----|----------|---------------|------------|
| AR1        |  –   |  –  |    –     | 1.00          | –          |
| RandomWalk |  –   |  –  |    –     | –             | –          |
| ...        |      |     |          |               |            |

## Repo layout

```
src/config.py            every tunable choice, in one place
src/data/                EconData + FRED clients (fetch-once parquet cache),
                         weekly dataset builder
src/models/              common fit/predict interface + 6 models
src/backtest.py          expanding-window walk-forward engine
src/evaluate.py          metric tables + Diebold–Mariano test
src/plots.py             all presentation figures
tests/test_backtest.py   synthetic-data proof of no look-ahead
run.py / Makefile        orchestration
```
