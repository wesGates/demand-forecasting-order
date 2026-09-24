"""
Render every figure to `figures/<notebook>/` as PNG, for review outside a
notebook.

    python -m src.render_figures

The notebooks are where the figures are read. This is where they get
regenerated after a change to `plots.py`, or to look at them all side by
side. One folder per notebook, numbered in the order the notebook draws them.
`figures/` is gitignored and the report embeds its own copies.

Careful with the bare module run. `render_order` needs the two-year (104
fold) run, and any method without a cached run on that layout gets fitted
from scratch. Call `render_all()` and `render_evaluation()` on their own when
that is all you need.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")  # file output only, never opens a window

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src import plots  # noqa: E402
from src.step1_problem import PROJECT_ROOT, STUDY_ITEMS, Config  # noqa: E402
from src.step2_data import MAJOR_EVENTS, load_panel  # noqa: E402
from src.step3_explore import (  # noqa: E402
    classification_cutoff,
    event_effects,
    series_stats,
)
from src.step4_models import MODELS  # noqa: E402

OUT = PROJECT_ROOT / "figures"
EXPLORE, EVALUATE, ORDER = OUT / "01_explore", OUT / "02_evaluate", OUT / "03_order"


def render_all(cfg: Config | None = None) -> list[str]:
    """Render the exploration figures (notebook 01). Returns the filenames written."""
    cfg = cfg or Config(item_ids=STUDY_ITEMS)
    EXPLORE.mkdir(parents=True, exist_ok=True)
    plots.use_style()

    item = cfg.item_ids[0]
    df = load_panel(cfg, verbose=False)
    stats = series_stats(df, cfg)
    cutoff = classification_cutoff(df, cfg)
    top_id, top_store = stats.iloc[0]["id"], stats.iloc[0]["store_id"]
    written: list[str] = []

    def save(fig: plt.Figure, name: str) -> None:
        fig.savefig(EXPLORE / name, bbox_inches="tight")
        plt.close(fig)
        written.append(f"{EXPLORE.name}/{name}")

    # 1. every store, with the held-out window marked and outliers flagged
    save(
        plots.plot_store_grid(df, item, stats, holdout_start=cutoff, flag_outliers=True),
        "1_grid.png",
    )

    # 2. demand classification
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    plots.plot_demand_class_map(stats, highlight_item=item, item_id=item, ax=ax)
    save(fig, "2_classmap.png")

    # 3a. seasonality at the busiest store, all years
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 2.9))
    plots.plot_seasonality(df, top_id, axes=axes)
    fig.suptitle(f"{item} at {top_store}", fontsize=11)
    save(fig, "3a_season_one_store.png")

    # 3b. seasonality across every store, each indexed to its own mean
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 2.9))
    plots.plot_seasonality_by_store(df, item, axes=axes)
    fig.suptitle(f"{item} — seasonal shape across stores", fontsize=11)
    save(fig, "3b_season_all_stores.png")

    # 4. autocorrelation and year-on-year overlay
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.4))
    plots.plot_acf(
        df, top_id, ax=axes[0], title=f"{item} at {top_store} — autocorrelation"
    )
    plots.plot_year_overlay(df, top_id, ax=axes[1])
    save(fig, "4_acf_overlay.png")

    # 5. relationships between variables
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    plots.plot_price_relationship(df, item, ax=axes[0])
    plots.plot_zero_rate(stats, item_id=item, ax=axes[1])
    save(fig, "5_relationships.png")

    # 5b. which calendar events move this item, the evidence for MAJOR_EVENTS.
    #     Measured before the classification cutoff so the choice of holiday
    #     features never sees the scored year.
    calendar = pd.read_csv(cfg.data_dir / "calendar.csv", parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    plots.plot_event_effects(event_effects(df, calendar, cutoff), major=MAJOR_EVENTS, ax=ax)
    save(fig, "6_event_effects.png")

    # 6. SNAP benefit days
    fig, ax = plt.subplots(figsize=(9, 3))
    plots.plot_snap_effect(df, item, stats, ax=ax)
    save(fig, "7_snap.png")

    return written


def render_evaluation(cfg: Config | None = None) -> list[str]:
    """
    The evaluation figures (notebook 02). Loads the cached walk-forward and
    draws FPP's evaluation plots from it. With every method cached this takes
    about a minute.
    """
    from src.step5_evaluate import run_walk_forward, score_folds

    cfg = cfg or Config(item_ids=STUDY_ITEMS)
    EVALUATE.mkdir(parents=True, exist_ok=True)
    plots.use_style()

    item = cfg.item_ids[0]
    df = load_panel(cfg, verbose=False)
    stats = series_stats(df, cfg)
    cutoff = classification_cutoff(df, cfg)
    origins = cfg.fold_origins(df["date"].max())
    top_id, top_store = stats.iloc[0]["id"], stats.iloc[0]["store_id"]

    predictions = run_walk_forward(df, cfg, progress=False)
    scores = score_folds(predictions)
    written: list[str] = []

    def save(fig: plt.Figure, name: str) -> None:
        fig.savefig(EVALUATE / name, bbox_inches="tight")
        plt.close(fig)
        written.append(f"{EVALUATE.name}/{name}")

    # 1. forecasts against actuals, busiest store (FPP §5.8)
    fig, ax = plt.subplots(figsize=(12.5, 3.6))
    plots.plot_forecast_folds(
        predictions, df, top_id, holdout_start=cutoff, origins=origins, ax=ax
    )
    save(fig, "1_forecasts_busiest_store.png")

    # 1b. the same, eight weeks around Thanksgiving and Christmas, readable
    fig, ax = plt.subplots(figsize=(11, 3.6))
    plots.plot_forecast_folds(
        predictions,
        df,
        top_id,
        origins=origins,
        window=("2015-11-09", "2016-01-03"),
        ax=ax,
    )
    save(fig, "1b_forecasts_busiest_store_holidays.png")

    # 2. error by horizon (FPP §5.10)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    plots.plot_rmsse_by_horizon(predictions, ax=ax)
    save(fig, "2_rmsse_by_horizon.png")

    # 3. residual diagnostics for each model at the busiest store (FPP §5.4)
    for model in MODELS:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
        plots.plot_residual_diagnostics(
            predictions, top_id, model, season=cfg.season, axes=axes
        )
        fig.suptitle(f"{model} at {top_store} — residual diagnostics", fontsize=11)
        save(fig, f"3_residuals_{model}.png")

    # 4. RMSSE by store, the study's own view
    fig, ax = plt.subplots(figsize=(10, 3.8))
    plots.plot_rmsse_by_store(scores, stats, item_id=item, ax=ax)
    save(fig, "4_rmsse_by_store.png")

    # 5. bias by store, the thing RMSSE cannot show
    fig, ax = plt.subplots(figsize=(10, 3.4))
    plots.plot_bias_by_store(scores, stats, item_id=item, ax=ax)
    save(fig, "5_bias_by_store.png")

    return written


def render_order(cfg: Config | None = None) -> list[str]:
    """
    The order-quantity figures (notebook 03; FPP §5.5, §5.9). Needs the
    two-year run. The first year calibrates each method's error quantiles and
    the second is judged.
    """
    from src.order import (
        calibrate,
        quantile_forecasts,
        score_quantiles,
        summarise_quantiles,
        weekly_totals,
    )
    from src.step5_evaluate import run_walk_forward

    cfg = cfg or Config(item_ids=STUDY_ITEMS)
    both = Config(**{**cfg.__dict__, "n_folds": 2 * cfg.n_folds})
    ORDER.mkdir(parents=True, exist_ok=True)
    plots.use_style()

    df = load_panel(both, verbose=False)
    stats = series_stats(df, cfg)
    top_id = stats.iloc[0]["id"]
    scored_from = cfg.holdout_start(df["date"].max()) - pd.Timedelta(days=1)

    weekly = weekly_totals(run_walk_forward(df, both, progress=False))
    scored = score_quantiles(
        quantile_forecasts(weekly, calibrate(weekly, scored_from), scored_from)
    )
    summary = summarise_quantiles(scored)
    written: list[str] = []

    def save(fig: plt.Figure, name: str) -> None:
        fig.savefig(ORDER / name, bbox_inches="tight")
        plt.close(fig)
        written.append(f"{ORDER.name}/{name}")

    # 1. does the ranking change with the cost asymmetry?
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    plots.plot_pinball_by_tau(summary, ax=axes[0])
    plots.plot_coverage_by_tau(summary, ax=axes[1])
    plots.merge_legends(fig)
    save(fig, "1_pinball_and_coverage.png")

    # 2. what the order would have been, week by week, at the busiest store
    fig, ax = plt.subplots(figsize=(12.5, 3.8))
    plots.plot_weekly_order_band(scored, top_id, "arima", ax=ax)
    save(fig, "2_weekly_order_band.png")

    return written


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # matplotlib font chatter
        names = render_all() + render_evaluation() + render_order()
    print(f"wrote {len(names)} figures to {OUT}")
    for n in names:
        print("  ", n)
