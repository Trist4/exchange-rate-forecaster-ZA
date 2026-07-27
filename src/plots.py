"""All report figures, saved to reports/figures/ at 150 dpi.

Each figure is its own function that RETURNS the matplotlib Figure, so
the notebooks can render them inline and main() can save them —
one code path, no notebook/script drift.
"""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.config import (
    DATA_PROCESSED,
    DATA_RESULTS,
    EVAL_END,
    EVAL_START,
    FEATURE_COLS,
    FIGURES,
    HORIZON_WEEKS,
)
from src.models import BENCHMARK_NAME

# Sized and scaled to stay readable when projected or embedded in slides.
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


def _save(fig: Figure, name: str) -> None:
    path = FIGURES / name
    fig.savefig(path, bbox_inches="tight")
    print(f"saved {path}")


def fig_rmse_bar(overall: pd.DataFrame) -> Figure:
    """(a) RMSE by model — numeric values printed on the bars so exact
    figures can be read without eyeballing bar heights."""
    tbl = overall.sort_values("rmse")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tab:orange" if m == BENCHMARK_NAME else "tab:blue" for m in tbl.index]
    bars = ax.bar(tbl.index, tbl["rmse"], color=colors)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=11)
    ax.set_ylabel("RMSE (ZAR per USD)")
    ax.set_title(f"Out-of-sample RMSE, {HORIZON_WEEKS}-week-ahead USDZAR "
                 f"({EVAL_START[:4]}–{EVAL_END[:4]})\norange = official AR(1) benchmark")
    ax.set_ylim(0, tbl["rmse"].max() * 1.15)  # headroom for the labels
    return fig


def fig_rmse_per_year(per_year: pd.DataFrame) -> Figure:
    """(b) RMSE per origin-year — shows WHEN each model wins/loses
    (e.g. calm 2024 vs choppy 2022 behave very differently)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for model in per_year.columns:
        style = dict(lw=3, color="tab:orange") if model == BENCHMARK_NAME else dict(lw=1.8)
        ax.plot(per_year.index, per_year[model], marker="o", label=model, **style)
    ax.set_xticks(per_year.index)
    ax.set_ylabel("RMSE (ZAR per USD)")
    ax.set_title("RMSE by forecast-origin year")
    ax.legend(ncol=3, fontsize=10)
    return fig


def fig_history(weekly: pd.DataFrame) -> Figure:
    """(c) Full USDZAR history with the evaluation window shaded, showing
    exactly which regimes the models were evaluated on."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(weekly.index, weekly["usdzar"], color="tab:blue", lw=1.2)
    ax.axvspan(pd.Timestamp(EVAL_START), pd.Timestamp(EVAL_END),
               color="tab:orange", alpha=0.15, label="evaluation window")
    ax.set_ylabel("ZAR per USD")
    ax.set_title("USDZAR, Friday-sampled weekly")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    return fig


def fig_predictors(weekly: pd.DataFrame, cols: list[str] | None = None) -> Figure:
    """(d) Small multiples of every feature — one glance shows scale,
    gaps and the Covid/2022 spikes each model gets to react to.
    `cols` lets ModelTester.ipynb plot a custom feature subset."""
    cols = cols if cols is not None else FEATURE_COLS
    n = len(cols)
    ncols = 2
    nrows = -(-n // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.4 * nrows), sharex=True)
    for ax, col in zip(axes.ravel(), cols):
        ax.plot(weekly.index, weekly[col], lw=0.9)
        ax.set_title(col, fontsize=11)
        ax.axvspan(pd.Timestamp(EVAL_START), pd.Timestamp(EVAL_END),
                   color="tab:orange", alpha=0.12)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle("Predictor series (orange band = evaluation window)")
    fig.tight_layout()
    return fig


def fig_forecast_vs_actual(results: pd.DataFrame, model_name: str) -> Figure:
    """(e) Best model's forecasts overlaid on realised USDZAR, plotted at
    the TARGET date (when the forecast was for, not when it was made)."""
    g = results[results["model"] == model_name].sort_values("target_date")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(g["target_date"], g["actual"], label="actual", color="black", lw=1.4)
    ax.plot(g["target_date"], g["forecast"], label=f"{model_name} forecast",
            color="tab:red", lw=1.1, alpha=0.85)
    ax.set_ylabel("ZAR per USD")
    ax.set_title(f"{model_name}: {HORIZON_WEEKS}-week-ahead forecasts vs actual")
    ax.legend()
    return fig


def fig_schematic() -> Figure:
    """(f) The expanding-window design as a picture: training bar grows,
    forecast dot sits h weeks past each origin. Purely illustrative."""
    fig, ax = plt.subplots(figsize=(9, 4))
    n_illustrated = 6
    for i in range(n_illustrated):
        y = n_illustrated - i
        train_end = 10 + 2 * i
        ax.barh(y, train_end, left=0, height=0.5, color="tab:blue", alpha=0.6)
        ax.plot(train_end, y, "k|", markersize=14)              # origin Friday
        ax.plot(train_end + 2, y, "o", color="tab:red")          # target, h ahead
        ax.annotate("", xy=(train_end + 2, y), xytext=(train_end, y),
                    arrowprops=dict(arrowstyle="->", color="tab:red"))
    ax.set_yticks(range(1, n_illustrated + 1))
    ax.set_yticklabels([f"origin {n_illustrated - i}" for i in range(n_illustrated)])
    ax.set_xticks([])
    ax.set_xlabel("time →")
    ax.set_title("Expanding-window walk-forward:\n"
                 "blue = training data (grows), | = Friday origin, red ● = 4-week-ahead target")
    ax.grid(False)
    return fig


def main() -> None:
    # Imported here (not at module top) so the notebook can import the
    # fig_* functions without needing results files to exist yet.
    from src.evaluate import load_results, overall_table, per_year_table

    weekly = pd.read_parquet(DATA_PROCESSED / "weekly.parquet")
    results = load_results()
    overall = overall_table(results)
    best_model = overall.index[0]  # lowest RMSE
    print(f"best model by RMSE: {best_model}")

    _save(fig_rmse_bar(overall), "a_rmse_by_model.png")
    _save(fig_rmse_per_year(per_year_table(results)), "b_rmse_per_year.png")
    _save(fig_history(weekly), "c_usdzar_history.png")
    _save(fig_predictors(weekly), "d_predictors.png")
    _save(fig_forecast_vs_actual(results, best_model), "e_forecast_vs_actual.png")
    _save(fig_schematic(), "f_expanding_window_schematic.png")
    plt.close("all")


if __name__ == "__main__":
    main()
