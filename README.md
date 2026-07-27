# exchange-rate-forecaster-ZA

**USDZAR 1-month-ahead forecasting — SES × Codera hackathon.**

Every Friday from Jan 2021 to Dec 2025 we pretend to "stand" on that Friday,
train our models only on data available up to that day, and forecast the
rand/dollar rate 4 weeks ahead. We do that for 261 Fridays, then measure who
forecast best. The official benchmark to beat is AR(1).

> New to the repo? **[DEVELOPERSETUP.MD](DEVELOPERSETUP.MD)** has from-scratch
> setup instructions and a file-by-file tour of the code.

---

## ✅ Where we are: the whole pipeline WORKS, end to end

Everything below is built, tested, and has already been run on real data.
`make all` reproduces the entire thing.

### Done — data
- [x] **SARB data via EconData**: USDZAR (EXCX135), gold (USD), 3m JIBAR,
      10y bond yield. Dataset ID (`MARKET_RATES`) and series keys were
      discovered via the API and are filled into `src/config.py`.
- [x] **US data via FRED** (no key needed): broad dollar index, VIX, US 3m
      yield, Brent oil.
- [x] All downloads **cached** as parquet files in `data/raw/` — fetch once,
      then everything runs offline.
- [x] **Platinum via local CSV** (`data/raw/plt_per_troy_ounce.csv`,
      daily USD/oz back to 1969) — not available from SARB or FRED, so it's
      a hand-downloaded file converted into the same cache by
      `src/data/local_csv.py`. *(TODO: record source URL in config.py.)*
- [x] **Weekly Friday table** built (`data/processed/weekly.parquet`):
      812 Fridays, 2010→2026, each row = one Friday with the exchange rate
      + 8 predictor features.

### Done — models (all share one simple fit/predict interface)
- [x] **RandomWalk** — "next month = today". The classic FX benchmark.
- [x] **RW+Drift** — today plus the long-run average weekly trend.
- [x] **AR1** — the official benchmark we're judged against.
- [x] **OLS** — regression of the 4-week move on our economic features.
- [x] **Ridge / Lasso** — same regression with penalties, tuned safely
      (cross-validation that never peeks past the forecast date).

### Done — backtest, evaluation, plots
- [x] **Expanding-window backtest engine**: 261 Friday origins × 6 models,
      refit at every origin, results in `data/results/forecasts.parquet`.
- [x] **Metrics** (`reports/*.csv`): overall + per-year RMSE, MAE,
      directional hit rate, and a Diebold–Mariano significance test vs AR1.
- [x] **All 6 presentation figures** saved in `reports/figures/`:
      - `a_rmse_by_model.png` — RMSE bar chart, numbers printed on bars (judging requirement ✓)
      - `b_rmse_per_year.png` — who wins in which year
      - `c_usdzar_history.png` — the rand's history, evaluation window shaded
      - `d_predictors.png` — small charts of every feature
      - `e_forecast_vs_actual.png` — best model's forecasts vs reality
      - `f_expanding_window_schematic.png` — diagram of how the backtest works
- [x] **Demo notebook** (`notebooks/01_exploration.ipynb`) that shows all
      tables and figures inline.
- [x] **No-cheating test suite** (`make test`, 4 tests passing): corrupts
      all data *after* a forecast date and proves forecasts don't change —
      i.e. no model can see the future.

### Current results (the honest headline)

| Model      | RMSE   | vs AR1 | Hit rate |
|------------|--------|--------|----------|
| **AR1** 🏆 | 0.5589 | 1.000  | 56%      |
| RandomWalk | 0.5600 | 1.002  | n/a      |
| Lasso      | 0.5674 | 1.015  | 46%      |
| RW+Drift   | 0.5677 | 1.016  | 46%      |
| Ridge      | 0.5892 | 1.054  | 45%      |
| OLS        | 0.5950 | 1.065  | 45%      |

(8 features incl. the US credit spread and platinum. A consistent pattern:
every feature we add leaves Lasso *exactly* unchanged — it shrinks the
newcomers to zero — while unregularised OLS, forced to fit them, drifts
further behind AR1. Textbook regularisation, live on our own data.)

Nothing beats AR(1) overall — this is the famous Meese–Rogoff result and it
is a *defensible* finding, not a failure. Our best story for the judges:
**Lasso and OLS beat AR(1) in the turbulent years (2022–2023)** when
fundamentals actually moved, and lose in calm years when "no change" rules.

---

## 🗺️ Roadmap: what's left for us to do

### Must do (before demo)
- [ ] **Commit and push everything** — the work is currently only on this laptop.
- [ ] **Codev setup**: follow DEVELOPERSETUP.MD, get `make test` passing on
      the second machine.
- [ ] **Daily token refresh**: the EconData API token in `.env` dies every
      24 h. Grab a fresh one from the portal each morning (only needed for
      re-*fetching* — everything else runs from the cache).
- [ ] **Everyone reads the code tour** in DEVELOPERSETUP §7–8. Judges test
      whether *each* team member can explain the code.
- [ ] **Build the slide deck** — the figures in `reports/figures/` are made
      to be dropped straight into slides.

### Should try (to actually beat AR1 — ideas, in rough order of promise)
- [ ] **Use the 10y yield**: we already download `sa_10y` but no feature
      uses it. A yield-curve slope (10y − 3m) or SA–US long spread is a
      cheap experiment: add one line in `build_dataset.py` + one name in
      `FEATURE_COLS`, rerun `make build backtest evaluate plots`.
- [ ] **Forecast combination**: average AR1 + Lasso. Combinations are the
      one trick that reliably helps in FX; it's a ~10-line new model.
- [ ] **Regime story**: lean into the per-year result — e.g. a model that
      switches toward fundamentals when VIX is high.
- [ ] **Sanity-check features**: eyeball `d_predictors.png` for anything
      weird (gaps, spikes) in the SARB series.

### Nice to have (only if time allows)
- [ ] Rolling (rather than expanding) training window as a robustness check.
- [ ] Horizon sensitivity: rerun with `HORIZON_WEEKS = 5` (one constant in
      `src/config.py`) to show our "1 month = 4 weeks" choice doesn't drive
      the conclusions.
- [ ] Polish notebook into the live-demo script.

---

## How to run it

```bash
source .venv/bin/activate   # env already set up on this machine (else see DEVELOPERSETUP.MD)
make all                    # fetch → build → backtest → evaluate → plots
make test                   # the no-look-ahead proof (4 tests)
```

Stages can run individually: `python run.py build`, `python run.py plots`, etc.
After code changes to data/models/backtest, rerun `make test` first.

## Method in five bullets (for the judges)

- **Friday grid**: daily data sampled as "last value on or before each
  Friday" — what a forecaster that day would truly have seen (holidays!).
- **"1 month" = 4 weeks** (`HORIZON_WEEKS = 4` in config; calendar months
  have 4 *or* 5 Fridays, which would mix horizons).
- **Strictly pseudo-real-time**: at each origin the data is truncated at
  that Friday and every model refit — enforced by one line in
  `src/backtest.py` and proven by the test suite.
- **Financial predictors, not macro releases**: market prices print
  instantly and are never revised; CPI/GDP have publication lags that would
  quietly cheat.
- **Judged metric**: RMSE vs AR(1) in ZAR-per-USD levels, plus the
  Diebold–Mariano test for statistical significance.
