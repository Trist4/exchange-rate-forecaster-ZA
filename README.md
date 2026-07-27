# exchange-rate-forecaster-ZA

**1-month-ahead USDZAR forecasting, benchmarked against AR(1).**
SES × Codera hackathon submission.

Every Friday from 1 Jan 2021 to 31 Dec 2025 we stand at that Friday, train
only on data observable up to that day, and forecast the rand/dollar rate
4 weeks ahead — 261 strictly out-of-sample forecasts per model, scored by
RMSE against the AR(1) benchmark.

## Setup & reproduction

The full data snapshot is committed in `data/raw/` (~1.4 MB), so the entire
pipeline reproduces offline — **no API credentials needed**:

```bash
python3 -m venv .venv && source .venv/bin/activate
make install                                   # deps + econdatapy (test PyPI)
# macOS/Homebrew only, if econdatapy errors on tkinter:
#   brew install python-tk@3.12   (match your Python version)

python run.py build backtest evaluate plots    # ~10 min, skips fetch
make test                                      # 4 tests: no-look-ahead proof
jupyter lab                                    # notebooks/ModelTester.ipynb
```

Re-downloading data (`python run.py fetch`) is only needed to refresh the
snapshot; it requires an EconData personal API token in `.env` (copy
`.env.example`; tokens are valid ~24 h).

## Data

| series | source |
|---|---|
| USDZAR (target), gold (USD), 3m JIBAR, 10y bond yield | SARB via EconData (`MARKET_RATES`, USDZAR = EXCX135) |
| Broad dollar index, VIX, US 3m & 10y yields, Baa yield, Brent | FRED (no key) |
| Platinum (USD/troy oz, daily since 1969) | [macrotrends.net](https://www.macrotrends.net/) export, committed as `data/raw/plt_per_troy_ounce.csv` |

Eight predictors are built from these (see `FEATURE_INFO` in `src/config.py`
for the variable-to-series mapping): 4-week momentum in gold, platinum,
Brent and the dollar index; 4-week changes in VIX and the US credit spread
(Baa − 10y); the SA–US 3-month rate spread; and a 52-week mean-reversion gap.

**Why financial predictors, not macro releases:** market prices print in
real time and are never revised, so using their Friday value at a Friday
origin is honestly pseudo-real-time. CPI/GDP-type series have publication
lags and revisions that would quietly leak future information.

## Method

- **Friday grid.** Daily series sampled as *last observation on or before*
  each Friday (holidays leave some Fridays without prints; taking the next
  day instead would be look-ahead).
- **"1 month" = 4 weeks** (`HORIZON_WEEKS` in config): calendar months have
  4 or 5 Fridays, which would mix horizons across origins.
- **Strictly out of sample.** At each origin the engine truncates the data
  at that Friday and refits *every* model on the expanding window — one
  line in `src/backtest.py` enforces it, and `tests/test_backtest.py`
  proves it: corrupting all post-origin data changes no forecast by a bit.
- **Models**: RandomWalk, RW+Drift, **AR1 (benchmark)**, OLS, Ridge, Lasso,
  SVM (RBF kernel — the only nonlinear model), and Lasso+RW (Lasso plus an
  in-window-estimated correction from its own latest residual). All
  hyperparameter tuning uses time-series CV inside the training window.
- **Evaluation**: overall and per-year RMSE / MAE / directional hit-rate,
  plus a Diebold–Mariano test (squared-error loss, autocorrelation-robust
  variance, Harvey small-sample correction) of every model against AR1.

## Results

261 Friday origins, errors in ZAR/USD levels. Full tables in `reports/`,
figures in `reports/figures/` (RMSE bar chart has values printed on bars).

| Model      | RMSE   | vs AR1 | Hit rate | DM p-value |
|------------|--------|--------|----------|------------|
| **AR1** 🏆 | 0.5589 | 1.000  | 56%      | —          |
| RandomWalk | 0.5600 | 1.002  | n/a*     | 0.71       |
| Lasso+RW   | 0.5638 | 1.009  | 46%      | 0.73       |
| Lasso      | 0.5674 | 1.015  | 46%      | 0.53       |
| RW+Drift   | 0.5677 | 1.016  | 46%      | 0.51       |
| Ridge      | 0.5892 | 1.054  | 45%      | 0.02       |
| OLS        | 0.5950 | 1.065  | 45%      | 0.01       |
| SVM        | 0.6390 | 1.144  | 48%      | 0.00       |

\* a no-change forecast calls no direction, so its hit rate is undefined.

**Findings.** Nothing beats AR(1) overall — the classic Meese–Rogoff
result, and no model is statistically distinguishable from the benchmark
except Ridge/OLS/SVM, which are significantly *worse*. The per-year story
is the interesting one: fundamentals models beat AR(1) in the turbulent
years (2022–23) and lose in calm ones (2024–25). Error grows monotonically
with model flexibility (benchmarks → Lasso → Ridge → OLS → SVM); the SVM
is the extreme case — best model of all in 2022, worst overall, with an
in-sample/out-of-sample RMSE ratio of 1.72 (textbook overfitting). Lasso
shrugged off every feature we added (credit spread, platinum) by shrinking
it to zero, while OLS deteriorated each time. Our best challenger,
Lasso+RW, learns that Lasso's misses slightly *reverse* at the 4-week
horizon (ρ ≈ −0.05) and fades the latest error — improving on plain Lasso
in all five years and closing to 0.9% of AR(1).

**Variable selection.** OLS confidence intervals (2.5–97.5%, see
`ModelTester.ipynb` Step 5) find three significant predictors — the
52-week mean-reversion gap (strongest, p = 0.002), Brent momentum (sign
opposite the oil-importer story, likely a global risk-on proxy) and the
SA–US rate spread (risk-premium sign, not naive carry) — yet Lasso's
cross-validation **drops all eight features at the final origin**.
In-sample significance did not survive out-of-sample validation, which is
the whole gap between fitting and forecasting in one result. The model's
95% forecast band, built only from errors observable at each origin,
covers 97.6% of realised outcomes — honestly calibrated, slightly
conservative.

**Experiments that did not work.**
- *Residual correction the other way around*
  (`notebooks/Residual-Correction.ipynb`): a Lasso trained on the macro
  features to predict the *residuals of the RW+drift benchmark*. It did not
  improve on the benchmarks; kept for reference.
- Adding the US credit spread and platinum as features (both left Lasso
  unchanged and made OLS/Ridge worse — evidence the feature set is
  saturated, retained for the regularisation comparison).

## Methodology & rigour

- **The out-of-sample guarantee is structural, not procedural.** Every model
  receives its data through one line in `src/backtest.py`
  (`train_df = df.loc[df.index <= origin]`), so no individual model can
  leak. `tests/test_backtest.py` then proves it empirically: all data after
  an origin is replaced with random garbage and every model's forecast at
  that origin must be bit-for-bit unchanged — run at three origins for all
  eight models on every `make test`.
- **No tuning leakage.** Ridge/Lasso/SVM hyperparameters are chosen by
  `TimeSeriesSplit` cross-validation strictly *inside* each training
  window; validation folds always postdate their training folds.
- **Statistics respect the data's structure.** Weekly 4-week-ahead
  forecasts overlap by 3 weeks, making forecast errors ~MA(3); the
  Diebold–Mariano test therefore uses an autocorrelation-robust long-run
  variance (h−1 autocovariances) with the Harvey small-sample correction,
  rather than assuming independent errors.
- **Real-time honesty about the data itself.** Predictors are daily
  financial prices precisely because they have no publication lag or
  revisions. One known caveat is documented rather than hidden: the Fed's
  H.10 dollar-index series is published weekly, so its same-Friday value
  is technically a few days early — immaterial at a 4-week horizon, but
  stated.
- **Determinism.** One global seed, a frozen committed data snapshot, and
  a cache-only modelling path mean every figure and table in this repo
  reproduces exactly on any machine.
- **Negative results are reported**, not discarded (see above): the
  reverse residual-correction, and features that added no value.

## Code quality

- **One source of truth, everywhere.** All tunables live in
  `src/config.py`; all metric definitions in `src/evaluate.py`; all figure
  code in `src/plots.py`/`src/diagnostics.py`. Notebooks *call* this code
  and never re-implement it, so notebook numbers cannot drift from the
  official results (`ModelTester.ipynb` on default settings reproduces
  `reports/metrics_overall.csv` exactly).
- **One generic backtest engine** serves the pipeline, the test suite and
  the notebooks — the code path that is tested is the code path that runs.
- **Small, typed, commented.** Plain functions over classes (the model
  interface is the one deliberate exception — eight models genuinely need
  polymorphism), type hints throughout, and comments that explain *why* a
  choice was made rather than what a line does.
- **Reproducible environment.** Pinless but isolated venv, `make install`
  covering the test-PyPI dependency safely (`--no-deps` so test builds of
  other packages can never be pulled in), and a fetch-once parquet cache
  so `make all` runs offline.

## Technical defence (anticipated questions)

- *Why is "1 month" 4 weeks?* Calendar months contain 4 or 5 Fridays; a
  calendar horizon would mix horizons across origins. It is one constant
  (`HORIZON_WEEKS`) and the conclusions do not hinge on it.
- *Which AR(1) is the benchmark?* A direct 4-step projection (OLS of the
  level in 4 weeks on the level today) — an AR(1) at the forecast
  horizon's own frequency, with no compounding of one-step estimation
  error. Its fitted slope is ≈0.99, i.e. near-random-walk, which the
  RandomWalk row confirms independently.
- *Why evaluate in levels?* The task defines RMSE in ZAR per USD. Models
  work in logs internally (symmetric percentage moves); forecasts are
  exponentiated before scoring.
- *Is the best model overfitting?* No — Lasso's in-sample/out-of-sample
  RMSE ratio is 1.04 (SVM, by contrast, is 1.72). Lasso's cross-validated
  alpha shrinks most coefficients to exactly zero; its forecasts carry
  ~1% of the variance of actual moves.
- *Why does nothing beat AR(1)?* This is the expected result
  (Meese–Rogoff, 1983, and its long empirical afterlife): at a 1-month
  horizon FX is close to a random walk. Our contribution is showing
  *when* fundamentals help — turbulent 2022–23 — and quantifying the cost
  of flexibility everywhere else.
- *Why do forecast lines trail the actual by a month?* Because every model
  forecasts ≈ no change, each plotted forecast equals the origin-Friday
  spot carried 4 weeks forward — the forecast curve is the actual curve
  shifted right. The lag is the visual form of the core finding, not a
  data-alignment error (errors are scored same-date; see
  `ModelTester.ipynb` Steps 6–7).

## Repo layout

```
src/config.py                every tunable choice + feature catalogue
src/data/                    EconData/FRED/local-CSV clients (parquet cache),
                             weekly Friday dataset builder
src/models/                  common fit/predict interface + 8 models
src/backtest.py              expanding-window walk-forward engine
src/evaluate.py              metric tables + Diebold–Mariano test
src/plots.py                 report figures
src/diagnostics.py           per-model analysis used by ModelTester
tests/test_backtest.py       synthetic-data proof of no look-ahead
run.py / Makefile            orchestration: fetch → build → backtest → evaluate → plots
notebooks/ModelTester.ipynb  interactive: any model vs AR1, any feature subset
notebooks/01_exploration.ipynb   renders all official tables and figures
notebooks/RandomWalk.ipynb   standalone random-walk prototype (self-contained)
notebooks/Residual-Correction.ipynb  negative result kept for reference
```

## Future work

Yield-curve slope features (10y − 3m; the series are already cached),
forecast combination (AR1 + Lasso averaging), regime-switching toward
fundamentals when volatility is high, rolling-window robustness checks,
and a calendar-month horizon sensitivity run (`HORIZON_WEEKS` is one
constant).
