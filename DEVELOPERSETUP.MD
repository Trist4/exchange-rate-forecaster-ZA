# Developer Setup Guide

Everything you need to go from a fresh clone to a running pipeline, plus a
tour of the codebase so you can explain any file to a judge (explainability
is a scored criterion — every one of us must be able to walk through all of
this).

---

## 1. Prerequisites

- **Python 3.11+** (the repo was built and tested on 3.12). Check with:
  ```bash
  python3 --version
  ```
- **git**, and access to this repo.
- **EconData credentials** (a `client_id` and `client_secret`) — needed for
  the SARB data. The FRED (US) data needs no credentials at all.

---

## 2. Create the virtual environment and install dependencies

From the repo root:

```bash
# 1. Create the venv (a folder called .venv, already git-ignored)
python3 -m venv .venv

# 2. Activate it — do this in EVERY new terminal before working
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows (PowerShell: .venv\Scripts\Activate.ps1)

# 3. Install the main dependencies
pip install -r requirements.txt

# 4. Install econdatapy — this one is SEPARATE on purpose (see note below)
pip install -i https://test.pypi.org/simple/ --no-deps econdatapy
```

Or, after activating the venv, steps 3–4 in one go:

```bash
make install
```

**Why is econdatapy installed separately with `--no-deps`?**
It is published on *test* PyPI, not real PyPI. If we added test PyPI as an
extra index for the whole `requirements.txt`, pip could silently install
broken test builds of *other* packages. So we install everything else from
real PyPI first, then pull only `econdatapy` from test PyPI, with `--no-deps`
so its dependencies (pandas, requests, pytz, pyjwt) still come from real
PyPI — they're already covered by `requirements.txt`.

### macOS + Homebrew Python only: tkinter

`econdatapy` imports `tkinter` at import time (it has a GUI login fallback),
and Homebrew's Python doesn't bundle Tk. If you see
`ModuleNotFoundError: No module named '_tkinter'`, run:

```bash
brew install python-tk@3.12        # match your Python minor version
```

(Standard python.org installers and most Linux distros include tkinter
already; on Debian/Ubuntu it's `sudo apt install python3-tk`.)

### Verify the install

```bash
python -c "from econdatapy import read; print('econdatapy OK')"
python -m pytest tests/ -q         # should print: 4 passed
```

The tests run on synthetic data, so they pass **before** any real data is
downloaded — if they pass, your environment is good.

---

## 3. Credentials (.env)

EconData uses a **personal API token** — a long JWT string starting with
`eyJ...` that you copy from the EconData portal (econdata.co.za) after
logging in. Tokens are **valid for ~24 hours**, so expect to paste a fresh
one each hackathon day.

```bash
cp .env.example .env
```

Open `.env` and set — the raw token, no quotes, no `Bearer` prefix:

```
ECONDATA_CREDENTIALS=eyJraWQiOiJYL1ZJ...the-whole-long-string...
```

Our client (`src/data/econdata_client.py`) detects the format: a raw token
is injected straight into econdatapy's auth layer (that's all the library's
own login would produce anyway), while a legacy `client_id;client_secret`
pair — if your account has one — still goes through the library's native
flow. If the token has expired you get a clear error telling you to fetch
a fresh one.

`.env` is git-ignored — **never commit credentials**. The code loads it
automatically via `python-dotenv`; you don't need to `export` anything.

---

## 4. Discovery: finding the SARB dataset ID and series keys

**Why this step exists:** we refuse to guess EconData identifiers in code.
The USDZAR series is EconData code **EXCX135** (dataflow: South African
Reserve Bank → Rates → Market Rates), but the *dataset ID* that contains it,
and the exact *series keys* (which may carry extra dimensions, e.g.
`EXCX135.B`), have to be looked up once, by a human, and pasted into config.
That human is you or Lusanda. Here's how:

```bash
source .venv/bin/activate

# List EVERY series key + label in a dataset:
python -m src.data.econdata_client <DATASET_ID>

# Same, but only rows containing some text (case-insensitive):
python -m src.data.econdata_client <DATASET_ID> EXCX135
python -m src.data.econdata_client <DATASET_ID> gold
```

`<DATASET_ID>` is the EconData ID of the SARB **Market Rates** dataset —
find it on the EconData portal (econdata.co.za) under South African Reserve
Bank → Rates → Market Rates; the dataset's ID is shown with the dataset.
If the first ID you try is wrong, the API returns an error — just try the
next candidate; discovery never writes anything, it only prints.

You are hunting for **five** series (run the filter with different words:
`EXCX`, `gold`, `platinum`, `JIBAR`, `bill`, `bond`, `yield`):

| config name | what it is                          | hint                    |
|-------------|-------------------------------------|-------------------------|
| `usdzar`    | USD/ZAR exchange rate               | code EXCX135            |
| `gold`      | gold price (USD/oz)                 | SA's top export         |
| `platinum`  | platinum price                      | top PGM export          |
| `sa_3m`     | 3-month JIBAR **or** 3m T-bill rate | the SA short rate       |
| `sa_10y`    | 10-year government bond yield       | the SA long rate        |

Copy each **exact key** as printed (the whole left-hand column entry).

## 5. Where to paste the IDs

Open **`src/config.py`** and fill in the two TODO blocks (around line 45):

```python
ECONDATA_DATASET_ID: str = "PASTE_DATASET_ID_HERE"

SARB_SERIES: dict[str, str] = {
    "usdzar":   "PASTE_EXACT_KEY_HERE",   # EXCX135...
    "gold":     "PASTE_EXACT_KEY_HERE",
    "platinum": "PASTE_EXACT_KEY_HERE",
    "sa_3m":    "PASTE_EXACT_KEY_HERE",
    "sa_10y":   "PASTE_EXACT_KEY_HERE",
}
```

Notes:
- Your `client_id`/`client_secret` go **only** in `.env` (step 3), never in
  config.py. Dataset ID + series keys go **only** in config.py. Nothing
  secret lives in config.py, so it's safe to commit.
- If you paste a code that matches *part* of a key (e.g. `EXCX135` when the
  full key is `EXCX135.B`), the fetcher accepts it **only if the match is
  unique**; if it's ambiguous or missing it raises an error telling you to
  re-run discovery. It never guesses.

## 6. Run the whole pipeline

```bash
make all          # = python run.py = fetch -> build -> backtest -> evaluate -> plots
```

Or stage by stage:

```bash
python run.py fetch      # downloads SARB + FRED series -> data/raw/*.parquet
python run.py build      # weekly Friday table -> data/processed/weekly.parquet
python run.py backtest   # walk-forward forecasts -> data/results/forecasts.parquet
python run.py evaluate   # prints metric tables, saves CSVs -> reports/
python run.py plots      # saves 6 figures -> reports/figures/
```

Fetching is **cache-once**: each series is stored as `data/raw/<name>.parquet`
and never re-downloaded while the file exists (delete the file, or call the
fetchers with `force=True`, to refresh). Everything downstream reads only the
cache, so after one fetch the whole pipeline runs offline and reproducibly.

The demo notebook (`notebooks/01_exploration.ipynb`, run `jupyter lab` from
the repo root) loads the cached outputs and renders all figures inline.

---

## 7. Tour of the codebase (what you must be able to explain)

**The task.** Forecast USDZAR 1 month ahead, forecasting every Friday from
2021-01-01 to 2025-12-31, strictly out-of-sample, benchmarked against AR(1).
We read "1 month" as **4 weeks** on a Friday grid (`HORIZON_WEEKS = 4` in
config — one constant to change if organisers say otherwise), because
calendar months have 4 *or* 5 Fridays and would mix horizons.

- **`src/config.py`** — every tunable decision in one file: horizon, eval
  window, data start (2010, so the expanding window has ~11 years of history
  at the first origin), random seed, the series IDs you pasted, and
  `FEATURE_COLS`. When a judge asks "where is that decided?" — point here.

- **`src/data/econdata_client.py` / `fred_client.py`** — thin download
  clients with the fetch-once parquet cache. FRED gives the US-side
  predictors (broad dollar index, VIX, US 3m yield, Brent); EconData/SARB
  gives USDZAR, gold, platinum, SA rates. *Why financial series and not
  macro releases like CPI:* market prices are observable the instant they
  print and are never revised, so using Friday t's value at origin t is
  honestly real-time; macro data has publication lags and revisions.

- **`src/data/build_dataset.py`** — resamples every daily series to a
  common **Friday grid** taking the *last observation on or before* each
  Friday (holidays mean some Fridays have no print; taking the next
  available day would be look-ahead). Builds the target `log_usdzar` (logs:
  symmetric % moves, and the random walk is exact in logs) and 7 features:
  4-week log changes of gold/platinum/Brent/dollar index, 4-week VIX change,
  SA-minus-US 3m rate spread (carry), and the gap of log USDZAR from its
  trailing 52-week mean (mean reversion). Every value in row t uses only
  data dated ≤ t.

- **`src/models/`** — one tiny interface (`fit(train_df)` /
  `predict(origin_row) -> log forecast`) and six implementations:
  - `RandomWalk` — forecast = today. The classic unbeatable FX benchmark.
  - `RandomWalkDrift` — today + 4 × average weekly drift.
  - `AR1Level` — **the official benchmark**: OLS of y_{t+4} on y_t (a
    "direct" h-step AR(1); one regression, no compounding of 1-step errors).
  - `FundamentalsOLS` — OLS of the 4-week log *change* on all features.
    We predict changes, not levels: FX levels are near unit-root and level
    regressions produce spurious fits.
  - `RidgeModel` / `LassoModel` — same regression, penalised; alpha tuned
    by `GridSearchCV` with `TimeSeriesSplit` **inside the training window
    only** (validation folds always come after their training folds, and
    the search never sees past the origin — no leakage).

- **`src/backtest.py`** — the one expanding-window walk-forward engine.
  For each Friday origin it truncates the table at the origin
  (`df.loc[df.index <= origin]` — *that line is the pseudo-real-time
  guarantee*), refits every model on the truncated data, forecasts 4 weeks
  ahead, and records origin/target_date/model/forecast/actual in both log
  and level units.

- **`src/evaluate.py`** — overall and per-origin-year RMSE / MAE /
  directional hit-rate per model (levels, ZAR per USD — the unit the brief
  judges), plus a hand-implemented **Diebold–Mariano test** of each model
  vs AR(1): squared-error loss, long-run variance with h−1 autocovariances
  (overlapping 4-week forecasts make the loss differential ~MA(3)), and the
  Harvey small-sample correction. Negative DM stat = model beats AR(1).

- **`src/plots.py`** — all six presentation figures at 150 dpi, including
  the RMSE bar chart **with numeric values printed on the bars** (explicit
  judging requirement). Each figure is a function returning the Figure, so
  the notebook and the saved PNGs share one code path.

- **`run.py` / `Makefile`** — orchestration only; no logic lives there.

## 8. The test suite (`tests/test_backtest.py`)

Run it any time with `make test` or `python -m pytest tests/ -v`.
It uses seeded synthetic data shaped exactly like the real weekly table, so
it needs no downloads and is fully deterministic. Two tests, four cases:

1. **`test_horizon_alignment`** — for a backtest at one origin, checks that
   `target_date` is exactly 4 Fridays after the origin, and the recorded
   "actual" really is the table's value at that target date.

2. **`test_no_lookahead[120|160|200]`** — the important one, at three
   different origins: run the backtest at an origin, then **corrupt every
   row after that origin** with huge random garbage and run it again. All
   six models' forecasts must be bit-for-bit identical (`==`, no tolerance);
   the test also verifies the corruption really reached the data. If anyone
   ever introduces leakage — in a model, a feature, or the engine — this
   fails loudly. It exercises the *same* `run_backtest` function used in
   production, not a copy.

If you touch anything in `src/models/`, `src/backtest.py` or
`src/data/build_dataset.py`, re-run the tests before pushing.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No module named '_tkinter'` | `brew install python-tk@3.12` (macOS) / `apt install python3-tk` (Linux) |
| `No module named 'pytz'` (or `jwt`) | `pip install -r requirements.txt` again — these must come from real PyPI |
| `ECONDATA_CREDENTIALS is not set` | Do step 3; make sure the file is named exactly `.env`, in the repo root |
| `Your EconData API token expired at ...` | Tokens last ~24h — log in to the portal, copy a fresh one into `.env` |
| An HTTP 401/Unauthorized from the API | Usually the same thing: stale token, or a stray space/newline pasted into `.env` |
| A Tk login window pops up | The env var is missing/empty — check `.env` exists and has no typo in the variable name |
| `SARB_SERIES entries not filled` | Do steps 4–5 (discovery, then paste into `src/config.py`) |
| `Missing cached series [...] in data/raw/` | Run `python run.py fetch` before `build` |
| Tests pass but `make all` fails at fetch | That's expected until steps 3–5 are done; FRED still fetches fine |
