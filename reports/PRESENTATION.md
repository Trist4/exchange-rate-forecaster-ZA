# Presentation layout — USDZAR 1-month-ahead forecasting (2–5 min)

Target: ~10 slides, 20–25 seconds each. One idea per slide; the numbers on
the slide are the script. Visuals marked **[VISUAL]** with exactly where to
get them. Split speakers wherever you like — every slide is self-contained.

---

## Slide 1 — Title (10 s)

- **Forecasting USDZAR one month ahead — can anything beat AR(1)?**
- Team name / members
- One-line spoiler as the subtitle: *"261 out-of-sample forecasts,
  8 models, one classic result — and the interesting part is when and why."*

**[VISUAL]** `reports/figures/c_usdzar_history.png` faded as the background
(the rand's history with the orange evaluation window).

---

## Slide 2 — Objective / problem statement (20 s)

- Task: forecast USDZAR **1 month ahead**, forecasting **every Friday**,
  1 Jan 2021 – 31 Dec 2025, strictly **out of sample**
- Score: **RMSE vs an AR(1) benchmark** (ZAR per USD)
- Our reading: "1 month" = 4 Fridays on a weekly grid → **261 forecast
  origins** per model (a calendar month has 4 *or* 5 Fridays and would mix
  horizons — one config constant if the organisers prefer otherwise)

**[VISUAL]** none needed — three bullets, large font.

---

## Slide 3 — Data acquisition (25 s)

- **SARB via EconData** (`MARKET_RATES`): USDZAR (EXCX135), gold (USD),
  3m JIBAR, 10y bond yield
- **FRED**: broad dollar index, VIX, US 3m/10y yields, Baa credit spread,
  Brent
- **macrotrends.net**: daily platinum since 1969 (not in SARB or FRED)
- Everything cached as parquet and **committed** → whole pipeline
  reproduces offline, bit-identical, no credentials
- WHY only market prices, no CPI/GDP: prices print in real time and are
  never revised — macro releases would quietly leak future information

**[VISUAL]** `reports/figures/d_predictors.png` (small multiples of all
8 features, evaluation window shaded).

---

## Slide 4 — Method: the time machine (25 s)

- **Expanding-window walk-forward**: stand on each Friday, delete the
  future, refit every model, forecast 4 weeks out, score against what
  actually happened — repeat 261 times
- The guarantee is **one line of code**
  (`train_df = df.loc[df.index <= origin]`) …
- … and it is **proven, not promised**: the test suite corrupts all
  post-origin data with garbage and requires every forecast to be
  bit-for-bit unchanged (`make test`, 4 passing)
- Ridge/Lasso/SVM hyperparameters tuned by time-series CV *inside* each
  training window — no tuning leakage either

**[VISUAL]** `reports/figures/f_expanding_window_schematic.png` (the
growing-blue-bars diagram). Optional inset: screenshot of the green
"forecast unchanged after corrupting the future -> True" output from
**ModelTester.ipynb Step 3(b)** (run with `MODEL_NAME = 'Lasso+RW'`).

---

## Slide 5 — The model zoo (20 s)

- Two **benchmarks**: RandomWalk ("no change"), RW+Drift
- **AR1** — the official yardstick (direct 4-step projection; fitted
  slope ≈ 0.99, i.e. near-random-walk)
- Four **fundamentals models** of increasing freedom: Lasso → Ridge →
  OLS → SVM (RBF kernel, the only nonlinear one)
- One **hybrid**: Lasso+RW — Lasso plus a correction from its own latest
  residual, with the persistence ρ *estimated*, not assumed
- All share one interface, all run through the same engine

**[VISUAL]** the 8-row model table from **ModelTester.ipynb Step 2**
(screenshot the markdown table), or retype it as a slide table.

---

## Slide 6 — Headline result (30 s) ★ the RMSE chart the brief requires

- **Nothing beats AR(1)**: RMSE 0.5589 vs RandomWalk 0.5600, Lasso+RW
  0.5638, … SVM 0.6390
- Diebold–Mariano: no model is statistically distinguishable from AR(1) —
  except Ridge/OLS/SVM, which are significantly **worse** (p = 0.02/0.006/0.005)
- Error rises **monotonically with model freedom**:
  benchmarks → Lasso → Ridge → OLS → SVM
- This is the Meese–Rogoff (1983) result reproduced on 2021–25 data — a
  defensible finding, not a failure

**[VISUAL]** `reports/figures/a_rmse_by_model.png` — RMSE bars with the
numbers printed on them (explicit brief requirement; AR1 bar is orange).

---

## Slide 7 — The story behind the headline: regimes (25 s)

- Fundamentals **win in turbulent years**: 2022 and 2023 (Lasso+RW beats
  AR1 by 1.8% and 3.5%) — **lose in calm 2024–25**
- The extreme case: **SVM was the best model of all in 2022** (0.678 vs
  AR1's 0.694) and the worst overall — in-sample/out-of-sample RMSE ratio
  **1.72**, textbook overfitting (Lasso: 1.03)
- Flexibility buys crisis performance and pays for it everywhere else;
  calm weeks outnumber crises, so AR(1) wins the war

**[VISUAL]** `reports/figures/b_rmse_per_year.png` (per-year RMSE lines,
AR1 in bold orange — point at 2022–23 vs 2024–25).

---

## Slide 8 — Variable selection: significance ≠ forecastability (25 s)

- OLS with 2.5–97.5% confidence intervals finds **3 significant
  predictors**: mean-reversion gap (p = 0.002, strongest), Brent momentum
  (p = 0.02, sign opposite the oil-importer story → global risk-on proxy),
  SA–US rate spread (p = 0.03, risk-premium reading)
- Yet Lasso's cross-validation **dropped all 8 features at every year-end
  refit — 40/40 coefficients exactly zero**
- In-sample significance did not survive out-of-sample validation; the
  models that respected that distinction ranked best
- Every feature we added (credit spread, platinum) left Lasso *unchanged*
  and made OLS *worse*

**[VISUAL]** screenshot of the **two coefficient tables from
ModelTester.ipynb Step 5** side by side: the OLS table (with the 2.5%/97.5%
CI columns) and the Lasso+RW table (every row "dropped"). That contrast IS
the slide.

---

## Slide 9 — Why forecasts "lag" reality + honest uncertainty (25 s)

- Every model forecasts ≈ no change → each plotted forecast ≈ the
  origin-Friday spot carried 4 weeks forward → the forecast curve is the
  actual curve **shifted one month right**
- Not a bug: errors are scored same-date; the lag is the *visual form* of
  unforecastability (predicted moves carry ~1–3% of actual move variance)
- Our 95% forecast band — built only from errors observable at each
  origin — covers **97.6%** of outcomes: honestly calibrated, slightly
  conservative

**[VISUAL]** screenshot of the **interactive chart from ModelTester.ipynb
Step 6** run with `MODEL_NAME = 'Lasso+RW'`, zoomed to **2023-01 → 2023-12**
(the selloff makes the lag and the shaded band clearly visible; hover
tooltip open on a date if you can catch it).

---

## Slide 10 — Conclusion & takeaways (25 s)

- At a 1-month horizon USDZAR is **near-unforecastable** — AR(1) stands,
  and we can prove our test was honest (no-look-ahead test, committed data
  snapshot, one-command reproduction)
- But *when* matters: **fundamentals add value precisely in crisis
  regimes** (2022–23) — the practical insight for a desk
- Best challenger: **Lasso+RW** (0.9% behind AR1, indistinguishable at
  p = 0.73) — it learned that forecast errors slightly *reverse*
  (ρ ≈ −0.05) and fades its own last miss
- Future: forecast combination (AR1+Lasso), regime switching on VIX,
  yield-curve features — all one config line away
- **Repo**: github.com/Trist4/exchange-rate-forecaster-ZA — `make all`
  reproduces every number shown

**[VISUAL]** `reports/figures/e_forecast_vs_actual.png` as a quiet
background, or just the repo link large.

---

## Screenshot checklist (what to capture, one sitting)

Set `MODEL_NAME = 'Lasso+RW'` in **ModelTester.ipynb**, Run All, then grab:

1. Step 2 model table → Slide 5
2. Step 3(b) "forecast unchanged -> True" output → Slide 4 (inset)
3. Step 5 OLS coefficient table (CI columns visible) → Slide 8 (left)
4. Step 5 Lasso+RW coefficient table (all "dropped") → Slide 8 (right)
5. Step 6 interactive chart, slider zoomed to 2023 → Slide 9

Static figures (already saved, 150 dpi): `a`, `b`, `c`, `d`, `e`, `f` in
`reports/figures/` → Slides 6, 7, 1, 3, 10, 4 respectively.

## Timing budget

10 + 20 + 25 + 25 + 20 + 30 + 25 + 25 + 25 + 25 s ≈ **3 min 50 s** — inside
the 2–5 min window with room for one breath per slide. If forced to cut to
~2.5 min: drop Slides 5 and 9 (fold the model list into Slide 6's chart and
the lag remark into Slide 7).
