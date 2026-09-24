"""
Step 2 - Gathering information (FPP §1.6, step 2).

Turns the three raw M5 files into one daily panel, one row per
store-item-date, with the calendar and price columns attached.

Five things in here are easy to get wrong.

1. Filter before melting. The full panel is 30,490 series x 1,941 days,
   about 59 million rows once reshaped. Subsetting first keeps a notebook
   responsive and changes nothing about the method.

2. SNAP flags are per state. The calendar has `snap_CA`, `snap_TX` and
   `snap_WI`. Each row reads the column for its own store's state.

3. Pre-launch zeros mean "not for sale". An item's row spans the whole
   history, including the years before the product was stocked. Left in,
   those zeros teach a model the item sells nothing and drag every rolling
   average down. A missing price is the reliable signal that the item was
   not being sold that week.

4. A closure day is a missing observation. Every store records zero sales
   on Christmas Day because the stores were shut. Left as a zero it damages
   the following week: the seasonal naive reads it, every rolling mean
   drops, and ETS takes it as a level shock. FPP §13.7 treats such days as
   missing. This loader fills them from the preceding weeks and flags the
   row (`closure`) so step 5 leaves it out of the score.

5. Holiday proximity is a calendar fact, known years ahead. The loader
   attaches `is_holiday`, `days_to_holiday` and `days_since_holiday` for the
   events that measurably move this item (`MAJOR_EVENTS`, chosen from the
   event-effect table in step 3). An on/off flag alone cannot learn the
   run-up before Christmas or Thanksgiving, which is as large as the day
   itself in this data.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.step1_problem import Config

# Bump when load_m5, trim_prelaunch_zeros or impute_closures changes what it
# produces. The cache key includes this, so old parquet files stop being
# served.
CACHE_VERSION = 3  # 3: closures imputed from preceding weeks only (2026-09-23)

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CAL_COLS = [
    "d",
    "date",
    "wm_yr_wk",
    "wday",
    "month",
    "year",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
]
SNAP_BY_STATE = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}

# Calendar events that measurably move the study item, from
# `step3_explore.event_effects` run on the data before the classification
# cutoff. An event is in if its day, or the two days before it, runs at least
# 15% away from the same-weekday baseline. Most of the other 23 M5 events
# (sporting, most religious, minor national days) sit within a few percent
# of baseline. Two exceptions, both left out: "Chanukah End" clears the bar
# but falls inside the Christmas run-up in most years, and Martin Luther
# King Day clears it at 1.19x and was simply missed until the 2026-09-23
# review. Adding it is an experiment for the registry.
MAJOR_EVENTS = (
    "Christmas",
    "Thanksgiving",
    "LaborDay",
    "IndependenceDay",
    "ValentinesDay",
    "NewYear",
    "Easter",
)

# Events on which the stores are shut. Sales read zero because nobody could
# buy anything. See point 4 in the module docstring.
CLOSURE_EVENTS = ("Christmas",)

# Proximity features are clipped here. Beyond a month "days to the next
# holiday" says nothing about demand, and an unclipped count would hand a
# tree model a second copy of day-of-year.
HOLIDAY_CLIP_DAYS = 30


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def holiday_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    """
    One row per calendar date with the holiday-proximity columns.

    `is_holiday`         1 on a MAJOR_EVENTS day.
    `days_to_holiday`    days until the next major event, 0 on the day itself.
    `days_since_holiday` days since the last major event, 0 on the day itself.
    `closure`            True on a CLOSURE_EVENTS day.

    Both counts are clipped at HOLIDAY_CLIP_DAYS. Where the calendar runs out
    before the next event they take the clip value; the last M5 date is more
    than a month from any major event, so nothing scored is affected. Two
    counts are used because one signed distance clipped at +-k cannot tell
    "exactly k days after" from "nothing nearby".

    Everything here is a fact about the calendar, known in advance for any
    target date, so it can be a feature without leaking.
    """
    cal = calendar[["date", "event_name_1", "event_name_2"]].copy()
    names = cal[["event_name_1", "event_name_2"]]
    is_major = names.isin(MAJOR_EVENTS).any(axis=1)
    is_closure = names.isin(CLOSURE_EVENTS).any(axis=1)

    dates = cal["date"].to_numpy()
    major_dates = dates[is_major.to_numpy()]
    # searchsorted gives, for each date, the index of the next event at or
    # after it.
    nxt = np.searchsorted(major_dates, dates, side="left")
    prv = np.searchsorted(major_dates, dates, side="right") - 1
    clip = np.timedelta64(HOLIDAY_CLIP_DAYS, "D")
    to_next = np.where(
        nxt < len(major_dates),
        major_dates[np.minimum(nxt, len(major_dates) - 1)] - dates,
        clip,
    )
    since_prev = np.where(prv >= 0, dates - major_dates[np.maximum(prv, 0)], clip)

    out = pd.DataFrame({"date": cal["date"]})
    out["is_holiday"] = is_major.astype("int8").to_numpy()
    out["days_to_holiday"] = np.minimum(
        (to_next / np.timedelta64(1, "D")).astype(int), HOLIDAY_CLIP_DAYS
    ).astype("int16")
    out["days_since_holiday"] = np.minimum(
        (since_prev / np.timedelta64(1, "D")).astype(int), HOLIDAY_CLIP_DAYS
    ).astype("int16")
    out["closure"] = is_closure.to_numpy()
    return out


def impute_closures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace sales on closure days with the same-weekday mean of the four
    preceding weeks (FPP §13.7, missing values).

    The replacement is deliberately dull. It is there so the following week's
    lags, rolling means and smoothing states see a normal day where the store
    happened to be shut. Nobody forecasts Christmas from it; the row stays
    flagged `closure` and step 5 leaves it out of every score.
    """
    df = df.copy()
    closed = df.index[df["closure"]]
    if len(closed) == 0:
        return df
    by_key = df.set_index(["id", "date"])["sales"]
    closed_keys = set(zip(df.loc[closed, "id"], df.loc[closed, "date"], strict=True))
    from src.features import holiday_window  # local import; features never imports the loader

    in_window = holiday_window(df).to_numpy(dtype=bool)
    busy = set(zip(df.loc[in_window, "id"], df.loc[in_window, "date"], strict=True))
    replacement = {}
    for sid, day in closed_keys:
        values = []
        # Preceding four weeks only, so nothing from later in the year leaks
        # into the fill. Known mistake: the first version averaged the four
        # weeks after as well, and Christmas 2015 then carried a slice of
        # January 2016 into every origin of the following month (FPP §5.10).
        # Days inside a holiday window are skipped, so the fill is an ordinary
        # same-weekday and not a run-up day.
        for k in (-4, -3, -2, -1):
            key = (sid, day + pd.Timedelta(days=7 * k))
            if key in by_key.index and key not in closed_keys and key not in busy:
                values.append(float(by_key[key]))
        if values:
            replacement[(sid, day)] = float(np.mean(values))
    keys = list(zip(df.loc[closed, "id"], df.loc[closed, "date"], strict=True))
    df.loc[closed, "sales"] = [replacement.get(k, np.nan) for k in keys]
    df["sales"] = df["sales"].fillna(0.0).astype("float32")
    return df


def load_m5(
    data_dir: str | Path = "data",
    item_ids: tuple[str, ...] = (),
    store_ids: tuple[str, ...] = (),
    dept_id: str | None = None,
    cat_id: str | None = None,
) -> pd.DataFrame:
    """
    Return a long daily panel, one row per (id, date).

    Every filter is optional and is applied before the reshape. Passing
    nothing loads the whole panel, which is slow and large.
    """
    data_dir = Path(data_dir)
    sales = pd.read_csv(data_dir / "sales_train_evaluation.csv")
    calendar = pd.read_csv(data_dir / "calendar.csv", parse_dates=["date"])
    prices = pd.read_csv(data_dir / "sell_prices.csv")

    # ---- filter before melting ------------------------------------------
    if item_ids:
        sales = sales[sales["item_id"].isin(item_ids)]
    if store_ids:
        sales = sales[sales["store_id"].isin(store_ids)]
    if dept_id is not None:
        sales = sales[sales["dept_id"] == dept_id]
    if cat_id is not None:
        sales = sales[sales["cat_id"] == cat_id]
    if sales.empty:
        raise ValueError("No rows left after filtering - check the subset in Config.")

    day_cols = [c for c in sales.columns if c.startswith("d_")]

    # ---- wide -> long ----------------------------------------------------
    df = sales.melt(
        id_vars=ID_COLS, value_vars=day_cols, var_name="d", value_name="sales"
    )

    # ---- calendar --------------------------------------------------------
    df = df.merge(calendar[CAL_COLS], on="d", how="left", validate="m:1")
    df = df.merge(holiday_calendar(calendar), on="date", how="left", validate="m:1")

    # Each row takes the SNAP flag for its own state. An unmapped state would
    # keep the 0 default, which looks like a real flag and is wrong, so the
    # loader refuses instead.
    unmapped = set(df["state_id"].unique()) - set(SNAP_BY_STATE)
    if unmapped:
        raise ValueError(
            f"No SNAP column mapped for state(s) {sorted(unmapped)}. "
            f"Add them to SNAP_BY_STATE rather than defaulting to 0."
        )
    df["snap"] = 0
    for state, col in SNAP_BY_STATE.items():
        mask = df["state_id"] == state
        df.loc[mask, "snap"] = df.loc[mask, col]
    df = df.drop(columns=list(SNAP_BY_STATE.values()))

    # ---- prices (weekly, fanned out to daily) ----------------------------
    # validate="m:1" matters more than it looks. A duplicate (store, item,
    # week) in the price file would give a store-item two rows for one date,
    # and every positional lag downstream would then point at the wrong day
    # with no error.
    df = df.merge(
        prices, on=["store_id", "item_id", "wm_yr_wk"], how="left", validate="m:1"
    )

    df["sales"] = df["sales"].astype("float32")
    return df.sort_values(["id", "date"]).reset_index(drop=True)


def trim_prelaunch_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop each series' rows from before the item was first offered for sale.

    Uses the first date with a price on file. Everything earlier goes,
    whatever the sales column says. How much this removes depends on the
    subset, so `load_panel` prints the figure per run.

    Known limitation: only the leading run of unpriced days is trimmed. An
    item withdrawn and later relisted keeps its mid-history gap, which reads
    as zero demand. Checked and not present in the current study items.
    """
    first_priced = (
        df[df["sell_price"].notna()]
        .groupby("id", observed=True)["date"]
        .min()
        .rename("launch_date")
    )

    # A series that was never priced has no launch date. Every date comparison
    # against NaT is False, so it would vanish without a trace. Warn instead.
    never_priced = sorted(set(df["id"].unique()) - set(first_priced.index))
    if never_priced:
        shown = never_priced[:5]
        more = " ..." if len(never_priced) > 5 else ""
        warnings.warn(
            f"{len(never_priced)} series have no price on record and are being "
            f"dropped entirely: {shown}{more}",
            stacklevel=2,
        )

    out = df.merge(first_priced, on="id", how="left")
    out = out[out["date"] >= out["launch_date"]]
    return out.drop(columns=["launch_date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Cached entry point, the one notebooks call
# --------------------------------------------------------------------------- #


def _cache_key(cfg: Config) -> str:
    """Stable hash of the subset and the build-logic version."""
    payload = repr((CACHE_VERSION, sorted(cfg.subset.items())))
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def load_panel(cfg: Config, use_cache: bool = True, verbose: bool = True) -> pd.DataFrame:
    """
    Load, join and trim the panel described by `cfg`, and cache the result.

    The CSV melt takes tens of seconds and the parquet round trip under one.
    Cached files live in `cfg.cache_dir`, which is gitignored. Bumping
    CACHE_VERSION or deleting that directory forces a rebuild.
    """
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.cache_dir / f"panel_v{CACHE_VERSION}_{_cache_key(cfg)}.parquet"

    if use_cache and path.exists():
        df = pd.read_parquet(path)
        if verbose:
            print(f"loaded cached panel: {path.name}  ({len(df):,} rows)")
        return df

    raw = load_m5(cfg.data_dir, **cfg.subset)
    df = impute_closures(trim_prelaunch_zeros(raw))
    assert_daily_grid(df)

    if verbose:
        dropped = len(raw) - len(df)
        print(
            f"built panel: {df['id'].nunique():,} series, {len(df):,} rows, "
            f"{df['date'].min().date()} -> {df['date'].max().date()}\n"
            f"  pre-launch rows dropped : {dropped:,} ({dropped / len(raw):.1%})\n"
            f"  closure days imputed    : {int(df['closure'].sum()):,}\n"
            f"  zero-sales share before : {(raw['sales'] == 0).mean():.1%}\n"
            f"  zero-sales share after  : {(df['sales'] == 0).mean():.1%}"
        )

    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)  # write then rename, so a killed build leaves no half-written panel
    return df


def assert_daily_grid(df: pd.DataFrame) -> None:
    """
    Every series must be one unbroken run of days.

    The lags, rolling windows, benchmarks and the RMSSE scale downstream are
    all positional (`sales[-7]` means a week ago). A missing day would shift
    all of them by one with no error. M5 has no gaps; a panel from a database
    might. FPP §13.7 says to fill or flag missing days first.
    """
    gaps = df.groupby("id", observed=True)["date"].agg(
        lambda s: int((s.sort_values().diff().dropna() != pd.Timedelta(days=1)).sum())
    )
    bad = gaps[gaps > 0]
    if len(bad):
        raise ValueError(
            f"{len(bad)} series have gaps in their daily dates (e.g. {bad.index[:3].tolist()}); "
            "the panel must be a complete daily grid - reindex and flag the missing days first"
        )


if __name__ == "__main__":
    cfg = Config(item_ids=("FOODS_3_120", "FOODS_3_681", "FOODS_3_282"))
    print(cfg.describe(), "\n")
    df = load_panel(cfg, use_cache=False)
    print("\nseries per item/store:")
    print(df.groupby(["item_id", "store_id"], observed=True).size().unstack().to_string())
