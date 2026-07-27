"""Per-model diagnostics for the model notebooks (LassoModel.ipynb etc.).

Design rule: notebooks NEVER re-implement fetching, backtesting or metrics.
They read the shared backtest output (data/results/forecasts.parquet) and
call the functions here, so every notebook shows numbers that agree with
the official pipeline — one source of truth, no drift between notebooks.

Everything takes `model_name` so the same notebook works for any model by
changing one string at the top.
"""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.config import EVAL_END, EVAL_START, FEATURE_COLS, HORIZON_WEEKS, TARGET_COL
from src.evaluate import diebold_mariano, load_results, overall_table
from src.models import BENCHMARK_NAME


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def model_slice(results: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """All scored forecasts of one model, ordered by origin."""
    g = results[results["model"] == model_name].sort_values("origin")
    if g.empty:
        raise ValueError(
            f"No forecasts for {model_name!r}. Available: "
            f"{sorted(results['model'].unique())}. Did the backtest run?"
        )
    return g


def summary(results: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """One-row headline: RMSE/MAE/hit-rate, ratio to AR1, DM test vs AR1."""
    tbl = overall_table(results)
    row = tbl.loc[[model_name]].copy()
    if model_name != BENCHMARK_NAME:
        g = model_slice(results, model_name).set_index("origin")
        b = model_slice(results, BENCHMARK_NAME).set_index("origin")
        common = g.index.intersection(b.index)
        stat, p = diebold_mariano(
            (g.loc[common, "forecast"] - g.loc[common, "actual"]).to_numpy(),
            (b.loc[common, "forecast"] - b.loc[common, "actual"]).to_numpy(),
        )
        row["dm_stat_vs_ar1"], row["dm_pvalue"] = stat, p
    return row


def direction_confusion(
    results: pd.DataFrame, model_name: str, normalise: bool = False
) -> pd.DataFrame:
    """2x2 'confusion matrix' for DIRECTION of the 4-week move.

    Rows = what the model called (rand weakens / strengthens vs the origin
    spot), columns = what actually happened. A forecast has no true class
    probabilities here, so direction is the natural classification view of
    a continuous FX forecast. No-change calls (RandomWalk) are excluded —
    they predict no direction to be right or wrong about.
    """
    g = model_slice(results, model_name)
    pred = np.sign(g["forecast"] - g["origin_spot"])
    act = np.sign(g["actual"] - g["origin_spot"])
    keep = pred != 0
    labels = {1.0: "weaker rand (USDZAR up)", -1.0: "stronger rand (USDZAR down)"}
    tbl = pd.crosstab(
        pred[keep].map(labels).rename("predicted"),
        act[keep].map(labels).rename("actual"),
        normalize="all" if normalise else False,
    )
    # Fixed row/col order so every model's matrix reads the same way.
    order = [labels[1.0], labels[-1.0]]
    return tbl.reindex(index=order, columns=order, fill_value=0)


def per_year_vs_benchmark(results: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Yearly RMSE of this model next to AR1, plus the ratio (<1 = winning)."""
    r = results[results["model"].isin([model_name, BENCHMARK_NAME])].copy()
    r["year"] = r["origin"].dt.year
    r["sq_err"] = (r["forecast"] - r["actual"]) ** 2
    tbl = np.sqrt(r.groupby(["year", "model"])["sq_err"].mean()).unstack()
    tbl[f"{model_name} / AR1"] = tbl[model_name] / tbl[BENCHMARK_NAME]
    return tbl


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_confusion(results: pd.DataFrame, model_name: str) -> Figure:
    """Directional confusion matrix as a heatmap with counts printed."""
    tbl = direction_confusion(results, model_name)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(tbl.to_numpy(), cmap="Blues")
    ax.set_xticks([0, 1], [c.replace(" (", "\n(") for c in tbl.columns])
    ax.set_yticks([0, 1], [i.replace(" (", "\n(") for i in tbl.index])
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted")
    total = tbl.to_numpy().sum()
    for i in range(2):
        for j in range(2):
            n = tbl.iat[i, j]
            ax.text(j, i, f"{n}\n({n / total:.0%})", ha="center", va="center",
                    color="white" if n > tbl.to_numpy().max() / 2 else "black")
    correct = np.trace(tbl.to_numpy())
    ax.set_title(f"{model_name}: direction of {HORIZON_WEEKS}-week move\n"
                 f"hit rate {correct / total:.1%} ({correct}/{total})")
    ax.grid(False)
    return fig


def fig_error_timeline(results: pd.DataFrame, model_name: str) -> Figure:
    """Forecast error over time, model vs AR1 — shows WHEN it wins/loses
    (the per-year story, but at full resolution)."""
    g = model_slice(results, model_name)
    b = model_slice(results, BENCHMARK_NAME)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(b["origin"], (b["forecast"] - b["actual"]).abs(),
            label=f"|error| {BENCHMARK_NAME}", color="tab:orange", lw=1.2)
    ax.plot(g["origin"], (g["forecast"] - g["actual"]).abs(),
            label=f"|error| {model_name}", color="tab:blue", lw=1.2)
    ax.set_ylabel("absolute error (ZAR)")
    ax.set_title(f"{model_name} vs {BENCHMARK_NAME}: absolute forecast error by origin")
    ax.legend()
    return fig


def fig_error_hist(results: pd.DataFrame, model_name: str) -> Figure:
    """Error distribution: bias shows as an off-centre hump."""
    g = model_slice(results, model_name)
    err = g["forecast"] - g["actual"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(err, bins=30, color="tab:blue", alpha=0.8)
    ax.axvline(0, color="black", lw=1)
    ax.axvline(err.mean(), color="tab:red", lw=1.5, ls="--",
               label=f"mean error {err.mean():+.3f} (bias)")
    ax.set_xlabel("forecast − actual (ZAR)")
    ax.set_title(f"{model_name}: error distribution")
    ax.legend()
    return fig


# ---------------------------------------------------------------------------
# Forecast-comparison overlay with a selectable window (used with the
# ipywidgets slider in ModelTester.ipynb Step 6)
# ---------------------------------------------------------------------------
def _window(g: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Filter one model's forecasts to target dates inside [start, end]."""
    if start is not None:
        g = g[g["target_date"] >= pd.Timestamp(start)]
    if end is not None:
        g = g[g["target_date"] <= pd.Timestamp(end)]
    return g


def fig_forecast_comparison(
    results: pd.DataFrame, model_name: str, start=None, end=None
) -> Figure:
    """Actual (black solid) vs AR1 (orange dashed) vs the chosen model
    (blue solid), all at TARGET dates — 'what did each say the rate would
    be on this day, and what was it'. Zooming into a window (via the
    notebook slider) makes the two forecasts distinguishable; at full
    range they both hug the actual, which is itself informative."""
    g = _window(model_slice(results, model_name), start, end)
    b = _window(model_slice(results, BENCHMARK_NAME), start, end)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(g["target_date"], g["actual"], color="black", lw=1.7, label="actual USDZAR")
    ax.plot(b["target_date"], b["forecast"], color="tab:orange", ls="--", lw=1.4,
            label=f"{BENCHMARK_NAME} forecast")
    ax.plot(g["target_date"], g["forecast"], color="tab:blue", lw=1.2, alpha=0.9,
            label=f"{model_name} forecast")
    ax.set_ylabel("ZAR per USD")
    span = (f"{g['target_date'].min().date()} to {g['target_date'].max().date()}"
            if len(g) else "empty window")
    ax.set_title(f"{HORIZON_WEEKS}-week-ahead forecasts vs reality — {span}")
    ax.legend()
    return fig


def window_metrics(
    results: pd.DataFrame, model_name: str, start=None, end=None
) -> pd.DataFrame:
    """Metric table for exactly the window shown in the comparison plot,
    recomputed on the fly: how do the two models score on just this slice
    of history? (Same metric code as the official evaluation.)"""
    from src.evaluate import _metrics  # same definitions as reports/*.csv

    r = results[results["model"].isin([model_name, BENCHMARK_NAME])]
    r = _window(r, start, end)
    if r.empty:
        return pd.DataFrame({"note": ["no scored forecasts in this window"]})
    tbl = r.groupby("model").apply(_metrics, include_groups=False)
    tbl["rmse_vs_ar1"] = tbl["rmse"] / tbl.at[BENCHMARK_NAME, "rmse"]
    return tbl.sort_values("rmse").round(4)


def attach_crosshair(fig: Figure) -> None:
    """Live readout for the comparison chart (needs the interactive ipympl
    backend, i.e. %matplotlib widget — does nothing useful on static PNGs).

    As the mouse moves over the axes, a dotted vertical guide snaps to the
    nearest plotted Friday and a box shows EVERY line's value on that date
    (actual, AR1, model) — so one glance gives the full comparison at any
    point, no squinting at pixel heights.
    """
    ax = fig.axes[0]
    lines = [ln for ln in ax.get_lines() if len(ln.get_xdata()) > 0]
    if not lines:
        return
    # Pre-convert each line's x values to matplotlib's float date units so
    # the mouse position (also float units) can be compared directly.
    xnums = [mdates.date2num(ln.get_xdata()) for ln in lines]

    guide = ax.axvline(ax.get_xlim()[0], color="grey", lw=0.8, ls=":", visible=False)
    box = ax.annotate(
        "", xy=(0.99, 0.02), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.9),
        visible=False,
    )

    def on_move(event) -> None:
        if event.inaxes is not ax or event.xdata is None:
            guide.set_visible(False)
            box.set_visible(False)
            fig.canvas.draw_idle()
            return
        # Snap to the nearest Friday on the first (actual) line.
        i = int(np.argmin(np.abs(xnums[0] - event.xdata)))
        snap_x = xnums[0][i]
        parts = [mdates.num2date(snap_x).strftime("%Y-%m-%d")]
        for ln, xn in zip(lines, xnums):
            j = int(np.argmin(np.abs(xn - snap_x)))
            parts.append(f"{ln.get_label():<14s} {ln.get_ydata()[j]:7.3f}")
        guide.set_xdata([snap_x, snap_x])
        guide.set_visible(True)
        box.set_text("\n".join(parts))
        box.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)


def window_table(
    results: pd.DataFrame, model_name: str, start=None, end=None, rows: int = 12
) -> pd.DataFrame:
    """The numbers behind the comparison plot, side by side (last `rows`
    target dates of the window).

    How to read the famous 'one-month lag' here: compare any forecast to
    the 'spot at origin' column — they nearly match. Each model basically
    carries the origin-Friday rate forward 4 weeks, so the forecast curve
    is the actual curve shifted one month right. The 'actual' column is
    what the rand then did instead.
    """
    g = _window(model_slice(results, model_name), start, end).set_index("target_date")
    cols = {
        "made on (origin)": g["origin"].dt.date,
        "spot at origin": g["origin_spot"],
    }
    if model_name != BENCHMARK_NAME:
        b = _window(model_slice(results, BENCHMARK_NAME), start, end).set_index("target_date")
        cols[f"{BENCHMARK_NAME} forecast"] = b["forecast"]
    cols[f"{model_name} forecast"] = g["forecast"]
    cols["actual"] = g["actual"]
    if model_name != BENCHMARK_NAME:
        cols[f"{BENCHMARK_NAME} |err|"] = (b["forecast"] - b["actual"]).abs()
    cols[f"{model_name} |err|"] = (g["forecast"] - g["actual"]).abs()
    tbl = pd.DataFrame(cols).tail(rows).round(3)
    tbl.index = tbl.index.date
    tbl.index.name = "target date"
    return tbl


# ---------------------------------------------------------------------------
# Summary statistics (brief requirement: obs counts for estimation and
# forecast periods)
# ---------------------------------------------------------------------------
def obs_counts(weekly: pd.DataFrame) -> pd.DataFrame:
    """Observation counts + basic USDZAR stats per period.

    'Estimation' = data before the first forecast origin (what the first
    model fit sees); the training set then GROWS through the forecast
    window because of the expanding-window design — by the last origin the
    models train on estimation + almost all of the forecast window.
    """
    parts = {
        "estimation (pre-2021)": weekly[weekly.index < EVAL_START],
        "forecast window (2021-25)": weekly[
            (weekly.index >= EVAL_START) & (weekly.index <= EVAL_END)
        ],
        "full weekly table": weekly,
    }
    return pd.DataFrame(
        {
            name: {
                "fridays": len(p),
                "first": p.index.min().date(),
                "last": p.index.max().date(),
                "usdzar mean": round(float(p["usdzar"].mean()), 2),
                "usdzar min": round(float(p["usdzar"].min()), 2),
                "usdzar max": round(float(p["usdzar"].max()), 2),
            }
            for name, p in parts.items()
        }
    ).T


# ---------------------------------------------------------------------------
# Overfitting / "lag" checks (why model and AR1 error curves look identical)
# ---------------------------------------------------------------------------
def error_correlation(results: pd.DataFrame, model_name: str,
                      other: str = BENCHMARK_NAME) -> float:
    """Correlation between two models' error series, matched by origin.

    Near 1.0 means the models make essentially the SAME errors — which
    happens when both forecast tiny moves, so both error series are just
    (minus) the actual 4-week move. That, not overfitting, is why the
    error-timeline chart shows two near-identical wiggly lines.
    """
    a = model_slice(results, model_name).set_index("origin")
    b = model_slice(results, other).set_index("origin")
    common = a.index.intersection(b.index)
    return float((a.loc[common, "forecast"] - a.loc[common, "actual"])
                 .corr(b.loc[common, "forecast"] - b.loc[common, "actual"]))


def error_autocorr(results: pd.DataFrame, model_name: str, max_lag: int = 6) -> pd.Series:
    """Autocorrelation of the (signed) error series by lag in weeks.

    The 'lag/wave' look of the error chart is mechanical: consecutive
    weekly origins forecast 4-week windows that OVERLAP by 3 weeks, so
    neighbouring errors share most of their outcome — theory says the
    errors behave like an MA(h-1) process: strong autocorrelation at lags
    1..3, roughly none from lag 4. This lets you verify that directly.
    """
    g = model_slice(results, model_name).set_index("origin")
    err = g["forecast"] - g["actual"]
    return pd.Series(
        {f"lag {k}w": round(float(err.autocorr(k)), 3) for k in range(1, max_lag + 1)},
        name=f"{model_name} error autocorrelation",
    )


def fig_move_scatter(results: pd.DataFrame, model_name: str) -> Figure:
    """Predicted vs actual 4-week log move. THE overfitting-or-not picture:
    a cloud hugging the horizontal axis means the model barely commits to
    any move (no overfit, little signal); points spread along the 45° line
    would mean real predictive power."""
    g = model_slice(results, model_name)
    pred = g["forecast_log"] - g["origin_log"]
    act = g["actual_log"] - g["origin_log"]
    fig, ax = plt.subplots(figsize=(6, 6))
    lim = float(act.abs().max()) * 1.1
    ax.axline((0, 0), slope=1, color="grey", ls="--", lw=1, label="perfect forecast (45°)")
    ax.axhline(0, color="black", lw=0.8)
    ax.scatter(act, pred, s=14, alpha=0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("actual 4-week log move")
    ax.set_ylabel("predicted 4-week log move")
    ax.set_title(f"{model_name}: predicted vs actual move\n"
                 f"sd(pred)/sd(actual) = {pred.std() / act.std():.2f},  "
                 f"corr = {pred.corr(act):+.2f}")
    ax.legend()
    return fig


def insample_vs_oos(weekly: pd.DataFrame, results: pd.DataFrame,
                    model_name: str, model_factory) -> pd.DataFrame:
    """The classic overfitting diagnostic: RMSE on data the model was fit
    on vs RMSE on data it never saw. A large gap (ratio >> 1) = overfit.

    Approximation, on purpose: we fit ONCE on the pre-2021 estimation
    sample and score it in-sample, then compare against the pipeline's
    true out-of-sample RMSE (which refits every week). Good enough to see
    whether a model memorises noise; simple enough to explain in a sentence.
    """
    train = weekly[weekly.index < EVAL_START]
    model = model_factory()
    model.fit(train)
    actual_log = train[TARGET_COL].shift(-HORIZON_WEEKS)
    sq = [
        (np.exp(model.predict(train.loc[t])) - np.exp(actual_log[t])) ** 2
        for t in train.index
        if not np.isnan(actual_log[t])
    ]
    ins = float(np.sqrt(np.mean(sq)))
    g = model_slice(results, model_name)
    oos = float(np.sqrt(((g["forecast"] - g["actual"]) ** 2).mean()))
    return pd.DataFrame(
        {"rmse": [ins, oos, oos / ins]},
        index=["in-sample (fit once, pre-2021)", "out-of-sample (pipeline)", "ratio"],
    ).round(4)


# ---------------------------------------------------------------------------
# Introspection for the tuned linear models (Ridge / Lasso)
# ---------------------------------------------------------------------------
def linear_coefs_over_time(
    weekly: pd.DataFrame, model_factory, origins: pd.DatetimeIndex
) -> pd.DataFrame:
    """Refit a Ridge/Lasso at selected origins and record coefficients + alpha.

    Uses the REAL model class from src/models (same fit code as the
    backtest), so what the notebook shows is exactly what the pipeline ran.
    Refitting is slow-ish (grid search per origin) — pass a handful of
    origins (e.g. one per year), not all 261.

    Coefficients are on STANDARDISED features, so magnitudes are directly
    comparable across features ("per one standard deviation move").
    """
    rows = {}
    for origin in origins:
        model = model_factory()
        model.fit(weekly.loc[weekly.index <= origin])
        reg = model._best.named_steps["reg"]
        rows[origin.date()] = pd.Series(
            dict(zip(FEATURE_COLS, reg.coef_)) | {"(alpha)": reg.alpha}
        )
    return pd.DataFrame(rows).T
