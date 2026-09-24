"""
Step 3 - Preliminary exploratory analysis (FPP §1.6, step 3).

FPP says to always start by graphing the data. Nothing here fits a model.
It answers the questions §1.6 asks before modelling (pattern, trend,
seasonality, outliers, relationships between variables) and adds the one
measurement this project is built around.

Classifying demand. "Sporadic and inconsistent" is a judgement until it is
measured. The standard measurement (Syntetos & Boylan) is two numbers.

  ADI  - average demand interval, days divided by days with a sale.
         How often does it sell? High ADI means sporadic.
  CV²  - squared coefficient of variation of the non-zero sale sizes.
         When it sells, how consistent is the amount? High CV² means
         inconsistent.

Cut each at its conventional threshold and you get four quadrants.

                 CV² < 0.49          CV² >= 0.49
    ADI <  1.32   smooth              erratic
    ADI >= 1.32   intermittent        lumpy

1.32 and 0.49 are conventions from one comparison of forecasting methods. A
series at ADI 1.31 is no different from one at 1.33. Treat the numbers as
continuous and the quadrants as labels of convenience; `plot_demand_class_map`
shows the whole cloud for that reason.

Availability comes before intermittency. A long block of zero sales usually
means the item was off the shelf. M5 has no inventory data, so stock-outs
and sporadic demand look the same in the sales column, and the pre-launch
price trim cannot catch them (these days do have a price on file). Left in,
they inflate ADI exactly like real intermittency. `max_zero_run` measures
this so a series can be screened, with a stated criterion, before anything
is modelled.

Everything here is computed on training data only. The class label is part
of the reported result, so no scored day may inform it.
`classification_cutoff` returns the first scored date and everything at or
after it is excluded.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.step1_problem import Config

# Syntetos-Boylan cut points. Conventions; see the module docstring.
ADI_CUT = 1.32
CV2_CUT = 0.49

# Availability screen. A zero run longer than this triggers a warning. A
# fast-moving item never posts a month of zeros, while the median FOODS_3
# series has an 83-day run. Past 30 days "off the shelf" is far more likely
# than "nobody wanted it".
MAX_ZERO_RUN_WARN = 30

# Every label `demand_class` can return, including the refusal case.
DEMAND_CLASSES = ("smooth", "erratic", "intermittent", "lumpy", "unclassifiable")


def classification_cutoff(df: pd.DataFrame, cfg: Config) -> pd.Timestamp:
    """
    The first date that will ever be scored, across every fold.

    Delegates to `Config.holdout_start`, so step 3 and step 5 agree on where
    the held-out window begins. Nothing on or after this date may enter a
    class label.
    """
    return cfg.holdout_start(df["date"].max())


def demand_class(adi: float, cv2: float) -> str:
    """Bucket one series from its ADI and CV². See the quadrant table above."""
    if not np.isfinite(adi) or not np.isfinite(cv2):
        return "unclassifiable"
    if adi < ADI_CUT:
        return "smooth" if cv2 < CV2_CUT else "erratic"
    return "intermittent" if cv2 < CV2_CUT else "lumpy"


def max_zero_run(sales: np.ndarray) -> int:
    """
    The longest unbroken stretch of zero-sales days, the availability screen.

    A fast-moving grocery item that sells every day should never post a
    two-week zero run. If it does, it was almost certainly unavailable.
    """
    best = current = 0
    for is_zero in sales == 0:
        current = current + 1 if is_zero else 0
        best = max(best, current)
    return best


def _series_row(sales: np.ndarray) -> dict[str, float]:
    """ADI and CV² for one series' sales vector."""
    n = len(sales)
    nonzero = sales[sales > 0]

    # CV² needs at least two sales to have a spread to measure. A series with
    # zero or one sale in the window gets no label.
    if len(nonzero) < 2:
        return {
            "adi": np.inf if len(nonzero) == 0 else n / len(nonzero),
            "cv2": np.nan,
            "max_zero_run": max_zero_run(sales),
        }

    adi = n / len(nonzero)
    cv2 = float((nonzero.std(ddof=1) / nonzero.mean()) ** 2)
    return {"adi": float(adi), "cv2": cv2, "max_zero_run": max_zero_run(sales)}


def series_stats(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    One row per series with size, sparsity, ADI, CV² and demand class.

    Computed on data before `classification_cutoff` only, so no scored day
    contributes to a label that later appears in the results table.
    """
    cutoff = classification_cutoff(df, cfg)
    train = df[df["date"] < cutoff]

    rows = []
    for series_id, g in train.groupby("id", sort=False, observed=True):
        sales = g["sales"].to_numpy(dtype=float)
        stats = _series_row(sales)
        rows.append(
            {
                "id": series_id,
                "item_id": g["item_id"].iloc[0],
                "store_id": g["store_id"].iloc[0],
                "state_id": g["state_id"].iloc[0],
                "n_days": len(sales),
                "mean_sales": float(sales.mean()),
                "zero_rate": float((sales == 0).mean()),
                **stats,
            }
        )

    out = pd.DataFrame(rows)
    out["demand_class"] = [
        demand_class(a, c) for a, c in zip(out["adi"], out["cv2"], strict=True)
    ]

    # The screen is only useful if it speaks up. A long dead block inflates ADI
    # like real intermittency, and a class label built on one measures shelf
    # availability. Say so before it gets modelled.
    flagged = out[out["max_zero_run"] > MAX_ZERO_RUN_WARN]
    if not flagged.empty:
        worst = flagged.sort_values("max_zero_run", ascending=False).head(5)
        detail = ", ".join(
            f"{r.item_id}@{r.store_id} ({int(r.max_zero_run)}d)"
            for r in worst.itertuples()
        )
        warnings.warn(
            f"{len(flagged)} of {len(out)} series have a zero-sales run longer than "
            f"{MAX_ZERO_RUN_WARN} days - their demand class likely reflects stocking "
            f"gaps, not customer behaviour. Worst: {detail}",
            stacklevel=2,
        )

    return out.sort_values(["item_id", "mean_sales"], ascending=[True, False])


def event_effects(
    df: pd.DataFrame, calendar: pd.DataFrame, cutoff: pd.Timestamp | None = None
) -> pd.DataFrame:
    """
    How much each calendar event moves sales relative to a same-weekday
    baseline. The evidence behind `step2_data.MAJOR_EVENTS`.

    For every event occurrence and every store, sales on the day (and on the
    two days before and the day after) are divided by the mean of the same
    weekday in the four weeks either side, skipping any week whose matching
    day is itself an event. The ratios are averaged over stores and years.
    Two-sided weeks are fine here because this is a measurement on the
    training period, not a forecast.

    A ratio of 1.70 means the day ran 70% above an ordinary same-weekday. A
    closure day (Christmas) is NaN on the day, since the stores were shut and
    the loader filled the value; its run-up is what counts. One table sorted
    by the size of the effect settles which holidays matter for this item.
    """
    events = pd.concat(
        [
            calendar.dropna(subset=["event_name_1"])[
                ["date", "event_name_1", "event_type_1"]
            ],
            calendar.dropna(subset=["event_name_2"])[
                ["date", "event_name_2", "event_type_2"]
            ].rename(
                columns={"event_name_2": "event_name_1", "event_type_2": "event_type_1"}
            ),
        ]
    ).sort_values("date")
    event_days = set(events["date"])
    # `cutoff` is the first scored day. Which events become features is a
    # modelling choice, and a choice made on the test period would be
    # selection on the test set (FPP §5.8). Pass `classification_cutoff`.
    if cutoff is not None:
        df = df[df["date"] < cutoff]
    wide = df.pivot(index="date", columns="id", values="sales")
    # A closure day's sales were filled by the loader (see step 2). There is
    # no demand to measure, so it must not show up here as "ratio 1.0".
    if "closure" in df:
        shut = df.loc[df["closure"], "date"].unique()
        wide.loc[wide.index.isin(shut)] = np.nan

    def baseline(day: pd.Timestamp) -> pd.Series | None:
        ref = [day + pd.Timedelta(days=7 * k) for k in (-4, -3, -2, -1, 1, 2, 3, 4)]
        ref = [d for d in ref if d in wide.index and d not in event_days]
        return wide.loc[ref].mean() if ref else None

    offsets = {"d-2": -2, "d-1": -1, "d0": 0, "d+1": 1}
    rows = []
    for _, ev in events.iterrows():
        if ev["date"] not in wide.index:
            continue
        rec = {
            "event": ev["event_name_1"],
            "type": ev["event_type_1"],
            "year": ev["date"].year,
        }
        for label, off in offsets.items():
            day = ev["date"] + pd.Timedelta(days=off)
            base = baseline(day) if day in wide.index else None
            if base is not None:
                rec[label] = float((wide.loc[day] / base).mean())
        rows.append(rec)

    table = (
        pd.DataFrame(rows)
        .groupby(["type", "event"])
        .agg(n_years=("year", "size"), **{k: (k, "mean") for k in offsets})
    )
    table["max_deviation"] = (table[list(offsets)] - 1).abs().max(axis=1)
    return table.sort_values("max_deviation", ascending=False)


def class_table(stats: pd.DataFrame) -> pd.DataFrame:
    """The item x store grid of demand classes, the study design at a glance."""
    return stats.pivot(index="item_id", columns="store_id", values="demand_class")


def summarise(stats: pd.DataFrame, cutoff: pd.Timestamp) -> str:
    """A short text summary, for the top of a notebook section."""
    counts = stats["demand_class"].value_counts()
    spread = (
        stats.groupby("item_id", observed=True)["demand_class"]
        .nunique()
        .rename("classes")
    )
    lines = [
        f"classification window ends {cutoff.date()} "
        f"({stats['n_days'].min():,}-{stats['n_days'].max():,} days per series)",
        f"series: {len(stats)}",
        f"longest zero run: {stats['max_zero_run'].min()}-{stats['max_zero_run'].max()} "
        f"days (availability screen)",
        "class counts: " + ", ".join(f"{k}={v}" for k, v in counts.items()),
        "classes spanned per item: " + ", ".join(f"{k}={v}" for k, v in spread.items()),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    from src.step1_problem import STUDY_ITEMS
    from src.step2_data import load_panel

    cfg = Config(item_ids=STUDY_ITEMS)
    df = load_panel(cfg, verbose=False)

    stats = series_stats(df, cfg)
    print(summarise(stats, classification_cutoff(df, cfg)), "\n")
    print(class_table(stats).to_string(), "\n")
    cols = [
        "item_id",
        "store_id",
        "mean_sales",
        "zero_rate",
        "max_zero_run",
        "adi",
        "cv2",
        "demand_class",
    ]
    print(stats[cols].round(3).to_string(index=False))
