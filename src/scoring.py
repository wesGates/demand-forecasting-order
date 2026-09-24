"""
Scoring and tables, from the long predictions table step 5 produces.

Kept apart from the harness on purpose. The predictions cache is keyed on the
harness module's code, and a change to how results are summarised must not
invalidate hours of fitting. Everything here reads predictions and writes
tables. Nothing here changes a forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.step4_models import BENCHMARKS

# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

FOLD_KEYS = ["id", "item_id", "store_id", "fold", "origin", "week_kind", "method", "kind"]


def score_folds(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    One row per series-fold-method with RMSE, MAE, bias and RMSSE.

    RMSSE is rmse / scale, where the scale came with the predictions. Every
    method in a series-fold shares that scale, which is why RMSSE cannot change
    who wins a fold, only how folds and stores compare.

    Closure days are dropped first, so a fold with one is scored on six days.
    """
    if "closure" in predictions:
        predictions = predictions[~predictions["closure"].astype(bool)]

    def one(g: pd.DataFrame) -> pd.Series:
        err = (g["forecast"] - g["actual"]).to_numpy(dtype=float)
        scale = float(g["scale"].iloc[0])
        # A fold is unscored when a forecast is missing or the scale is zero or
        # NaN. It is counted in n_unscored and dropped from averages and pairs.
        # The old code scored these as infinity, which made the mean RMSSE
        # infinite for any method with one such fold.
        finite = bool(np.isfinite(err).all())
        scorable = finite and np.isfinite(scale) and scale > 0
        rmse = float(np.sqrt(np.mean(err**2))) if finite else np.nan
        return pd.Series(
            {
                "rmse": rmse,
                "mae": float(np.mean(np.abs(err))) if finite else np.nan,
                "bias": float(np.mean(err)) if finite else np.nan,
                "rmsse": rmse / scale if scorable else np.nan,
                "unscored": not scorable,
                "fallback": bool(g["fallback"].any()) if "fallback" in g else False,
                "n_days": len(g),
                # the store's level that week, what "busiest first" sorts on
                "mean_actual": float(np.mean(g["actual"])),
            }
        )

    return (
        predictions.groupby(FOLD_KEYS, sort=True, observed=True)
        .apply(one, include_groups=False)
        .reset_index()
    )


def _weeks(scores: pd.DataFrame, week_kind: str | None) -> pd.DataFrame:
    """Filter to one kind of week, or keep all when `week_kind` is None."""
    if week_kind is None:
        return scores
    if week_kind not in ("normal", "holiday"):
        raise ValueError("week_kind must be None, 'normal' or 'holiday'")
    return scores[scores["week_kind"] == week_kind]


def rmsse_by_store(scores: pd.DataFrame, week_kind: str | None = None) -> pd.DataFrame:
    """
    Stores down, methods across, mean RMSSE over folds. The headline table.

    `week_kind` restricts it to "normal" or "holiday" folds. None pools both.
    """
    scores = _weeks(scores, week_kind)
    table = scores.pivot_table(
        index="store_id", columns="method", values="rmsse", aggfunc="mean"
    )
    # Stores by volume, busiest first, and methods by overall RMSSE. Volume is
    # the mean actual level. Known mistake: an earlier version sorted by RMSE
    # ascending, which put the quietest store first under a busiest-first label.
    store_order = (
        scores.groupby("store_id", observed=True)["mean_actual"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    method_order = table.mean(axis=0).sort_values().index
    return table.reindex(index=store_order, columns=method_order)


def win_rates(scores: pd.DataFrame, week_kind: str | None = None) -> pd.DataFrame:
    """
    For every method, the share of series-folds where it beat each benchmark.

    Rows are methods, columns are benchmarks, values are fractions. 0.5 means
    no better than the benchmark. A benchmark against itself is left blank. A
    win rate says how often and never by how much, so it sits beside RMSSE.

    `week_kind` restricts it to "normal" or "holiday" folds. None pools both.
    """
    scores = _weeks(scores, week_kind)
    wide = scores.pivot_table(
        index=["id", "fold"], columns="method", values="rmsse", aggfunc="first"
    )
    out = {}
    for bench in BENCHMARKS:
        if bench not in wide:
            continue
        out[bench] = {}
        for m in wide.columns:
            if m == bench:
                out[bench][m] = np.nan
                continue
            pair = wide[[m, bench]].dropna()  # the same pairs improvement_over uses
            out[bench][m] = float((pair[m] < pair[bench]).mean()) if len(pair) else np.nan
    return pd.DataFrame(out).sort_index()


# The benchmark the headline claims are made against. Seasonal naive is what
# an orderer's default screen shows (this day last week). It is also the RMSSE
# denominator, which makes "improvement over it" and "1 - RMSSE" the same
# number seen two ways. The 28-day moving average is the harder benchmark and
# gets reported beside it as the stress check.
REFERENCE_BENCHMARK = "seasonal_naive"


def improvement_over(
    scores: pd.DataFrame,
    benchmark: str = REFERENCE_BENCHMARK,
    week_kind: str | None = None,
) -> pd.DataFrame:
    """
    Per method, how often and by how much it beat one benchmark, fold by fold.

    `win_rate` is the share of series-folds with lower RMSSE than the
    benchmark. The improvement columns are the per-fold percentage reduction
    in RMSSE, as a mean and as quartiles, because a mean can hide a quarter of
    folds that got worse. The benchmark's own row is dropped.
    """
    scores = _weeks(scores, week_kind)
    wide = scores.pivot_table(
        index=["id", "fold"], columns="method", values="rmsse", aggfunc="first"
    )
    if benchmark not in wide:
        raise ValueError(f"{benchmark!r} is not among the scored methods")
    rows = []
    for m in wide.columns:
        if m == benchmark:
            continue
        pair = wide[[m, benchmark]].dropna()
        # A benchmark can score exactly zero on a fold, a week of zero sales
        # forecast as zero on an intermittent item. A percentage improvement
        # over zero is undefined. Those folds still count toward the win rate
        # (nothing beats a zero) but not toward the quartiles, and n_undefined
        # says how many were dropped.
        undefined = pair[benchmark] <= 0
        gain = (pair[benchmark] - pair[m])[~undefined] / pair[benchmark][~undefined] * 100
        rows.append(
            {
                "method": m,
                "win_rate": float((pair[m] < pair[benchmark]).mean()),
                "mean_improvement_pct": float(gain.mean()),
                "q1_pct": float(gain.quantile(0.25)),
                "median_pct": float(gain.median()),
                "q3_pct": float(gain.quantile(0.75)),
                "n_folds": len(pair),
                "n_undefined": int(undefined.sum()),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("median_pct", ascending=False)
        .reset_index(drop=True)
    )


def summarise(scores: pd.DataFrame) -> pd.DataFrame:
    """
    One row per method. Mean and median RMSSE over all series-folds, mean bias,
    the mean RMSSE on normal and holiday folds with the count of each, and the
    unscored and fallback counts. Sorted best first. this is the table to
    quote, and quote both week columns.
    """
    overall = scores.groupby(["method", "kind"], observed=True).agg(
        rmsse_mean=("rmsse", "mean"),
        rmsse_median=("rmsse", "median"),
        bias_mean=("bias", "mean"),
        n_folds=("rmsse", "size"),
        n_unscored=("unscored", "sum"),
        n_fallback=("fallback", "sum"),
    )
    by_kind = scores.pivot_table(
        index=["method", "kind"],
        columns="week_kind",
        values="rmsse",
        aggfunc=["mean", "size"],
        observed=True,
    )
    for week in ("normal", "holiday"):
        overall[f"rmsse_{week}"] = by_kind.get(("mean", week), np.nan)
        overall[f"n_{week}"] = by_kind.get(("size", week), 0)
    return overall.reset_index().sort_values("rmsse_mean").reset_index(drop=True)
