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
| Platinum (USD/troy oz, daily since 1969) | local CSV (`data/raw/plt_per_troy_ounce.csv`) |

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

**Experiments that did not work.**
- *Residual correction the other way around*
  (`notebooks/Residual-Correction.ipynb`): a Lasso trained on the macro
  features to predict the *residuals of the RW+drift benchmark*. It did not
  improve on the benchmarks; kept for reference.
- Adding the US credit spread and platinum as features (both left Lasso
  unchanged and made OLS/Ridge worse — evidence the feature set is
  saturated, retained for the regularisation comparison).

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
