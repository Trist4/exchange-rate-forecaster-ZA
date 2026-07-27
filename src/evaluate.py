"""Evaluation: RMSE/MAE/hit-rate tables + Diebold-Mariano test vs AR(1).

All error metrics are computed on the LEVEL exchange rate (ZAR per USD)
because that is the unit the task defines the RMSE comparison in; models
forecast logs internally but we exponentiate in the backtest output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.config import DATA_RESULTS, HORIZON_WEEKS, REPORTS
from src.models import BENCHMARK_NAME


def load_results() -> pd.DataFrame:
    df = pd.read_parquet(DATA_RESULTS / "forecasts.parquet")
    # Only origins whose outcome has been observed can be scored.
    return df.dropna(subset=["actual"])


def _metrics(g: pd.DataFrame) -> pd.Series:
    err = g["forecast"] - g["actual"]
    # Directional hit: did we call the SIGN of the move from the origin
    # spot correctly? Relevant because a desk trading on the forecast
    # cares about direction before magnitude. Forecasts that
    # predict exactly no change (the RandomWalk, by construction) call no
    # direction at all, so they are excluded rather than scored as misses
    # — its hit_rate shows as NaN, which is the truthful answer.
    pred_dir = np.sign(g["forecast"] - g["origin_spot"])
    actual_dir = np.sign(g["actual"] - g["origin_spot"])
    directional = pred_dir != 0
    return pd.Series(
        {
            "rmse": float(np.sqrt((err**2).mean())),
            "mae": float(err.abs().mean()),
            "hit_rate": float((pred_dir == actual_dir)[directional].mean())
            if directional.any()
            else np.nan,
            "n": int(len(g)),
        }
    )


def overall_table(results: pd.DataFrame) -> pd.DataFrame:
    tbl = results.groupby("model").apply(_metrics, include_groups=False)
    # Relative RMSE < 1 means "beats the official AR(1) benchmark".
    tbl["rmse_vs_ar1"] = tbl["rmse"] / tbl.at[BENCHMARK_NAME, "rmse"]
    return tbl.sort_values("rmse")


def per_year_table(results: pd.DataFrame) -> pd.DataFrame:
    """RMSE by origin-year (origin year, not target year: a forecast belongs
    to the moment it was made)."""
    r = results.assign(year=results["origin"].dt.year)
    return (
        r.groupby(["year", "model"])
        .apply(_metrics, include_groups=False)
        .reset_index()
        .pivot(index="year", columns="model", values="rmse")
    )


def diebold_mariano(
    e_model: np.ndarray, e_bench: np.ndarray, horizon: int = HORIZON_WEEKS
) -> tuple[float, float]:
    """DM test with squared-error loss and the Harvey small-sample fix.

    H0: equal predictive accuracy. NEGATIVE statistic => the model's
    squared errors are smaller than the benchmark's.

    The loss differential d_t = e_model^2 - e_bench^2 is autocorrelated by
    construction: consecutive weekly 4-step forecasts share 3 weeks of
    outcome, making d_t roughly MA(h-1). So the variance of d-bar uses a
    truncated (rectangular) long-run variance with h-1 autocovariances —
    the estimator in the original Diebold-Mariano (1995) paper.
    """
    d = e_model**2 - e_bench**2
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    lrv = float((dc**2).mean())  # gamma_0
    for lag in range(1, horizon):
        lrv += 2.0 * float((dc[lag:] * dc[:-lag]).mean())
    if lrv <= 0 or n <= horizon:  # degenerate sample — no verdict
        return np.nan, np.nan
    dm = dbar / np.sqrt(lrv / n)
    # Harvey, Leybourne & Newbold (1997) correction: the raw DM statistic
    # is oversized in modest samples at h>1; scale it and use Student-t.
    harvey = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    stat = float(harvey * dm)
    p_value = float(2 * stats.t.sf(abs(stat), df=n - 1))
    return stat, p_value


def dm_table(results: pd.DataFrame) -> pd.DataFrame:
    """DM statistic + p-value of every model against the AR(1) benchmark,
    matched origin-by-origin so both error series cover identical dates."""
    bench = results[results["model"] == BENCHMARK_NAME].set_index("origin")
    rows = []
    for name, g in results.groupby("model"):
        if name == BENCHMARK_NAME:
            continue
        g = g.set_index("origin")
        common = g.index.intersection(bench.index)
        e_model = (g.loc[common, "forecast"] - g.loc[common, "actual"]).to_numpy()
        e_bench = (bench.loc[common, "forecast"] - bench.loc[common, "actual"]).to_numpy()
        stat, p = diebold_mariano(e_model, e_bench)
        rows.append({"model": name, "dm_stat": stat, "p_value": p, "n": len(common),
                     "better_than_ar1": stat < 0 if not np.isnan(stat) else None})
    return pd.DataFrame(rows).set_index("model").sort_values("dm_stat")


def main() -> None:
    results = load_results()
    overall = overall_table(results)
    per_year = per_year_table(results)
    dm = dm_table(results)

    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    print("\n=== Overall metrics (level RMSE/MAE in ZAR, rmse_vs_ar1 < 1 beats benchmark) ===")
    print(overall)
    print("\n=== RMSE by origin year ===")
    print(per_year)
    print("\n=== Diebold-Mariano vs AR(1) (negative stat = model better) ===")
    print(dm)

    overall.to_csv(REPORTS / "metrics_overall.csv")
    per_year.to_csv(REPORTS / "metrics_per_year.csv")
    dm.to_csv(REPORTS / "diebold_mariano.csv")
    print(f"\nsaved tables to {REPORTS}/")


if __name__ == "__main__":
    main()
