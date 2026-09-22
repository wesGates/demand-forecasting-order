"""
From forecast to order quantity (FPP §5.5, §5.9).

Everything before this module produces a *point* forecast and scores it with
a symmetric error. Neither is what a replenishment decision needs.

**The order is a weekly total, not a daily point.** A Sunday order covers
Monday to Sunday, so the error that reaches the shelf is the sum of seven
forecasts minus the sum of seven actuals. Daily errors partly cancel inside a
week, and the daily RMSSE never sees that they did. `weekly_totals` builds the
quantity the order actually is.

**Ordering the mean stocks out half the time.** A point forecast is the
centre of what might happen; demand lands above it about as often as below.
The quantity to order is a *quantile*: the level that covers demand with a
chosen probability τ. Which τ is an economic question, not a statistical one
- the critical fractile τ = Cu / (Cu + Co), understock cost over the sum of
both. Above 0.5 when running out costs more than holding (ambient grocery);
below 0.5 when holding costs more (fresh produce that goes in the bin). M5
items are anonymised, so this study cannot know τ for its item, and reports a
range instead - and whether the method ranking changes across it.

**Where the quantiles come from.** None of the nine methods produces a
distribution, and XGBoost's point forecast is one number. FPP §5.5's route
works for all of them alike: a method's own past errors are its uncertainty.
For each store and method, the weekly errors from a *calibration year* give
an empirical error distribution; the τ-quantile forecast for a later week is
the point forecast plus the τ-quantile of those errors. A method with tighter
errors earns a smaller add-on. That is the whole reward for accuracy, and a
biased method is corrected automatically - the mean error is inside the
distribution being shifted by.

**Calibration weeks must precede scored weeks.** Estimating the quantile on
the weeks it is then judged on gives 90% coverage by construction. The run
that feeds this module scores two years; the first calibrates and the second
is judged. Every holiday appears in both, which a split of a single year
could not give.

**Pinball loss** (FPP §5.9, the quantile score) is the metric, with the book's
factor of two so that τ = 0.5 equals the absolute error:

    2 τ (y - q)        if y >= q   (under-forecast: shortfall)
    2 (1 - τ) (q - y)  if y <  q   (over-forecast: surplus)

At τ = 0.9 a unit of shortfall costs nine times a unit of surplus. The
asymmetry the decision cares about is inside the metric, not outside it. A
symmetric metric cannot see an asymmetric cost, and which way it misleads
depends on economics it has no access to.

Pinball losses at different τ are on different scales and must not be
averaged across τ. Within one τ they are in units, so the report divides by
the store's mean weekly sales to make a 100-unit store and a 15-unit store
comparable - the same reason RMSSE exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The service levels reported. Symmetric around 0.5 so the perishable and
# the ambient direction are both covered, plus the 0.9 an ambient item would
# typically use.
TAUS = (0.3, 0.5, 0.7, 0.9)

WEEK_KEYS = ["id", "item_id", "store_id", "fold", "origin", "week_kind", "method", "kind"]


def weekly_totals(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    One row per series-fold-method: the week's actual and forecast totals.

    Closure days are dropped first, as in `score_folds`, so both totals cover
    the same days. `error` is actual minus forecast: **positive is a
    shortfall**, the sign a replenishment decision cares about.
    """
    p = predictions
    if "closure" in p:
        p = p[~p["closure"].astype(bool)]
    out = (
        p.groupby(WEEK_KEYS, observed=True)
        .agg(
            actual=("actual", "sum"),
            forecast=("forecast", "sum"),
            n_days=("actual", "size"),
            scale=("scale", "first"),
        )
        .reset_index()
    )
    out["error"] = out["actual"] - out["forecast"]
    return out


def calibrate(weekly: pd.DataFrame, scored_from: pd.Timestamp, taus=TAUS) -> pd.DataFrame:
    """
    Per (series, method, τ): the empirical τ-quantile of weekly errors over
    the calibration folds - every fold whose origin is before `scored_from`.

    Returns `offset`, the add-on to a point forecast, and `n_calib`, how many
    weeks it rests on.
    """
    c = weekly[weekly["origin"] < scored_from]
    rows = []
    for (sid, method), g in c.groupby(["id", "method"], observed=True):
        for tau in taus:
            rows.append(
                {
                    "id": sid,
                    "method": method,
                    "tau": tau,
                    "offset": float(g["error"].quantile(tau)),
                    "n_calib": len(g),
                }
            )
    return pd.DataFrame(rows)


def calibrate_expanding(
    weekly: pd.DataFrame, scored_from: pd.Timestamp, taus=TAUS
) -> pd.DataFrame:
    """
    Like `calibrate`, but the calibration window grows: each scored fold's
    offset uses every fold before it, including earlier scored folds. This is
    what a live system would do - recalibrate as errors accumulate. Returns
    one row per (series, method, fold, τ).
    """
    rows = []
    for (sid, method), g in weekly.groupby(["id", "method"], observed=True):
        g = g.sort_values("origin")
        errors = g["error"].to_numpy()
        origins = g["origin"].to_numpy()
        for i, origin in enumerate(origins):
            if origin < scored_from:
                continue
            past = errors[:i]
            for tau in taus:
                rows.append(
                    {
                        "id": sid,
                        "method": method,
                        "fold": int(g["fold"].iloc[i]),
                        "tau": tau,
                        "offset": float(np.quantile(past, tau)),
                        "n_calib": len(past),
                    }
                )
    return pd.DataFrame(rows)


def quantile_forecasts(
    weekly: pd.DataFrame, offsets: pd.DataFrame, scored_from: pd.Timestamp
) -> pd.DataFrame:
    """
    The scored weeks with a τ-quantile forecast `q` for every τ in `offsets`:
    the point forecast plus that series-method's offset. `offsets` may be
    fixed (from `calibrate`) or per fold (from `calibrate_expanding`).
    """
    s = weekly[weekly["origin"] >= scored_from]
    keys = ["id", "method"] + (["fold"] if "fold" in offsets else [])
    out = s.merge(offsets, on=keys, how="inner")
    out["q"] = out["forecast"] + out["offset"]
    return out


def pinball(actual, q, tau) -> np.ndarray:
    """FPP §5.9's quantile score, with the factor of two (τ = 0.5 gives |error|)."""
    actual, q, tau = (np.asarray(x, dtype=float) for x in (actual, q, tau))
    diff = actual - q
    return np.where(diff >= 0, 2 * tau * diff, 2 * (1 - tau) * (-diff))


def score_quantiles(qf: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-week quantile scores to the output of `quantile_forecasts`:
    `pinball` (units), `pinball_rel` (pinball over the store's mean weekly
    sales in the scored period), `covered` (demand fell at or below q),
    `shortfall` and `surplus` (units).
    """
    qf = qf.copy()
    qf["pinball"] = pinball(qf["actual"], qf["q"], qf["tau"])
    level = qf.groupby("id", observed=True)["actual"].transform("mean")
    qf["pinball_rel"] = qf["pinball"] / level
    qf["covered"] = qf["actual"] <= qf["q"]
    qf["shortfall"] = np.maximum(qf["actual"] - qf["q"], 0)
    qf["surplus"] = np.maximum(qf["q"] - qf["actual"], 0)
    return qf


def summarise_quantiles(
    scored: pd.DataFrame, week_kind: str | None = None
) -> pd.DataFrame:
    """
    Per method and τ: mean relative pinball loss, achieved coverage, and the
    mean shortfall and surplus in units per week. Coverage should sit near τ;
    the gap between them is how well the calibration transferred from one
    year to the next.
    """
    if week_kind is not None:
        scored = scored[scored["week_kind"] == week_kind]
    return (
        scored.groupby(["method", "kind", "tau"], observed=True)
        .agg(
            pinball_rel=("pinball_rel", "mean"),
            coverage=("covered", "mean"),
            shortfall=("shortfall", "mean"),
            surplus=("surplus", "mean"),
            n_weeks=("covered", "size"),
        )
        .reset_index()
    )


def ranking_by_tau(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Methods down, τ across, relative pinball in the cells, best first at
    τ = 0.5. The table that answers "does the model ranking change with the
    cost asymmetry?"
    """
    table = summary.pivot_table(index="method", columns="tau", values="pinball_rel")
    order = (
        table[0.5].sort_values().index
        if 0.5 in table
        else table.mean(axis=1).sort_values().index
    )
    return table.reindex(order)


if __name__ == "__main__":
    from src.step1_problem import STUDY_ITEMS, Config
    from src.step2_data import load_panel
    from src.step5_evaluate import run_walk_forward

    # Two years of folds: the first calibrates, the second is judged. The
    # scored year is exactly the layout the point-forecast tables use.
    scored = Config(item_ids=STUDY_ITEMS)
    both = Config(item_ids=STUDY_ITEMS, n_folds=2 * scored.n_folds)
    panel = load_panel(both, verbose=False)
    scored_from = scored.holdout_start(panel["date"].max()) - pd.Timedelta(days=1)

    predictions = run_walk_forward(panel, both, progress=False)
    weekly = weekly_totals(predictions)
    print(
        f"{len(weekly):,} series-fold-method weeks; scored from origin {scored_from.date()}"
    )

    print(
        "\n=== weekly-total bias (forecast - actual, units/week) and share of weeks under-forecast ==="
    )
    w = weekly[weekly["origin"] >= scored_from].copy()
    w["under"] = w["error"] > 0
    print(
        w.groupby("method", observed=True)
        .agg(
            bias=("error", lambda e: -e.mean()),
            sd=("error", "std"),
            under=("under", "mean"),
        )
        .sort_values("sd")
        .round(2)
        .to_string()
    )

    offsets = calibrate(weekly, scored_from)
    scored_q = score_quantiles(quantile_forecasts(weekly, offsets, scored_from))
    summary = summarise_quantiles(scored_q)
    print(
        "\n=== relative pinball loss by method x tau (fixed calibration year), best at 0.5 first ==="
    )
    print(ranking_by_tau(summary).round(3).to_string())
    print("\n=== achieved coverage by method x tau (target = tau) ===")
    print(
        summary.pivot_table(index="method", columns="tau", values="coverage")
        .round(2)
        .to_string()
    )
    for week in ("normal", "holiday"):
        print(f"\n=== relative pinball, {week} weeks ===")
        print(ranking_by_tau(summarise_quantiles(scored_q, week)).round(3).to_string())

    expanding = calibrate_expanding(weekly, scored_from)
    scored_x = score_quantiles(quantile_forecasts(weekly, expanding, scored_from))
    print("\n=== relative pinball, expanding calibration window ===")
    print(ranking_by_tau(summarise_quantiles(scored_x)).round(3).to_string())
