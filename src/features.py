"""
Feature engineering.

One rule for the whole module. Every feature for a target day must be
computable from data available at the forecast origin T.

We forecast days T+1 .. T+h from one origin T, written `ŷ_{T+h|T}` in FPP.
A row for target day T+h may use
  - sales up to and including day T,
  - anything known in advance for T+h, like the calendar, SNAP dates and
    the horizon h.

Getting this wrong is leakage. The validation scores look excelent and then
collapse in production. Known mistake: it happened once on this project, so
`assert_no_leakage` now checks the dates on every fold.

Origin-based lags. Every row in a fold shares the one origin, the way a
replenishment run does (stand on Sunday, order for Monday through Sunday).
The lag columns are therefore identical across the h rows of a fold, and
`horizon` is what tells them apart, so `horizon` has to be a feature. The
other scheme, shifting each target day's features by its own horizon, is
also leak-free but forecasts each day from a different origin and throws
away up to six days of data the store would actually have.

The lags come from the autocorrelation in step 3. The weekly harmonics
(7, 14, 21, 28) dominate and lag 1 is strong on its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Lags counted back from the origin T. Picked from the ACF in step 3.
LAGS = (1, 2, 3, 7, 14, 21, 28)

# Rolling windows ending at T, for the recent level and its spread.
ROLL_WINDOWS = (7, 28, 56)

# Same-weekday history, the mean of the last k occurrences of the target's
# weekday at or before T. This is the learned version of the seasonal naive
# and usually the strongest single feature in daily retail.
DOW_WINDOWS = (4, 8)


# --------------------------------------------------------------------------- #
# Calendar, known in advance, so safe for any target date
# --------------------------------------------------------------------------- #


def calendar_features(rows: pd.DataFrame, use_price: bool = False) -> pd.DataFrame:
    """
    Features known in advance for the target date.

    None of these can leak. A calendar does not depend on what was sold, and
    in this dataset the shelf price is set a week ahead. Price is opt-in via
    `Config.use_price`; that comment says why it is off for the study item.
    """
    out = pd.DataFrame(index=rows.index)
    date = rows["date"]

    out["dow"] = date.dt.dayofweek
    out["is_weekend"] = (out["dow"] >= 5).astype("int8")
    out["day_of_month"] = date.dt.day
    out["month"] = date.dt.month
    out["day_of_year"] = date.dt.dayofyear
    out["snap"] = rows["snap"].astype("int8")
    out["is_event"] = rows["event_name_1"].notna().astype("int8")
    # Holiday proximity, from step 2's calendar columns. `is_event` above
    # fires on all 30 calendar events. These three cover only the events that
    # measurably move the item and let a model learn the run-up and the
    # hangover as well as the day. The run-up to Christmas in this data is
    # larger than most holidays' own day.
    out["is_holiday"] = rows["is_holiday"].astype("int8")
    out["days_to_holiday"] = rows["days_to_holiday"].astype("int16")
    out["days_since_holiday"] = rows["days_since_holiday"].astype("int16")
    if use_price:
        out["sell_price"] = rows["sell_price"].astype("float32")
    return out


# --------------------------------------------------------------------------- #
# History. Everything here must end at the origin
# --------------------------------------------------------------------------- #


# The holiday-affected window in days before and after a major event. Step 5's
# normal/holiday fold split and the loader's closure fill use the same
# numbers, so "holiday-affected" means one thing everywhere.
HOLIDAY_WINDOW_BEFORE, HOLIDAY_WINDOW_AFTER = 2, 1


def holiday_window(frame: pd.DataFrame) -> pd.Series:
    """True for rows inside the holiday window around a major event."""
    return (frame["days_to_holiday"] <= HOLIDAY_WINDOW_BEFORE) | (
        frame["days_since_holiday"] <= HOLIDAY_WINDOW_AFTER
    )


def history_features(history: pd.DataFrame, target_dates: pd.Series) -> pd.DataFrame:
    """
    Summaries of one series' sales, all ending at the forecast origin.

    `history` holds one series, sorted by date, with only dates at or before
    the origin T. The caller makes that cut and `assert_no_leakage` verifies
    it. `target_dates` are the days being forecast and only the same-weekday
    features look at them.

    Returns one row per target date. The lag and rolling columns describe the
    origin and so repeat across those rows, which is why `horizon` is also a
    feature.
    """
    sales = history["sales"].to_numpy(dtype=float)
    n = len(sales)
    out = pd.DataFrame(index=target_dates.index)

    # Lags counted back from the origin. lag_1 is the origin day itself.
    for k in LAGS:
        out[f"lag_{k}"] = sales[-k] if n >= k else np.nan

    # Level and spread over windows ending at the origin.
    for w in ROLL_WINDOWS:
        window = sales[-w:] if n >= 1 else np.array([])
        out[f"roll_mean_{w}"] = window.mean() if len(window) else np.nan
        out[f"roll_std_{w}"] = window.std(ddof=1) if len(window) > 1 else np.nan

    # Trend. Is the recent level above or below the longer-run level?
    out["trend_7_56"] = out["roll_mean_7"] / out["roll_mean_56"].replace(0, np.nan) - 1

    # same-weekday history, the learned seasonal naive. This one varies per
    # target day because each target day has its own weekday.
    hist_dow = history["date"].dt.dayofweek.to_numpy()
    target_dow = target_dates.dt.dayofweek.to_numpy()
    for k in DOW_WINDOWS:
        values = []
        for d in target_dow:
            same = hist_dow == d
            values.append(sales[same][-k:].mean() if same.any() else np.nan)
        out[f"dow_mean_{k}"] = values

    return out


def build_fold_features(
    history: pd.DataFrame,
    targets: pd.DataFrame,
    origin: pd.Timestamp,
    use_price: bool = False,
) -> pd.DataFrame:
    """
    The feature matrix for one fold of one series.

    `history` is already cut at `origin` and `targets` are the days being
    forecast. The returned frame carries `horizon` (1..h), which separates
    otherwise identical rows.
    """
    # The date check. History ends at or before the origin and every target
    # is after it, or this raises.
    assert_no_leakage(history, targets, origin)

    feats = pd.concat(
        [
            calendar_features(targets, use_price=use_price),
            history_features(history, targets["date"]),
        ],
        axis=1,
    )
    feats["horizon"] = (targets["date"] - origin).dt.days.astype("int16")
    return feats


# --------------------------------------------------------------------------- #
# The safeguard
# --------------------------------------------------------------------------- #


def assert_no_leakage(
    history: pd.DataFrame, targets: pd.DataFrame, origin: pd.Timestamp
) -> None:
    """
    Refuse to build features if anything reaches past the forecast origin.

    This proves the boundary by comparing dates. A statistical test can only
    say "this looks suspicious"; this says which row used data from after T,
    with the date. Raises ValueError if the history extends past the origin
    or any target is at or before it.
    """
    if len(history):
        latest = history["date"].max()
        if latest > origin:
            raise ValueError(
                f"LEAK: history reaches {latest.date()}, past origin {origin.date()}."
            )
    if len(targets):
        earliest = targets["date"].min()
        if earliest <= origin:
            raise ValueError(
                f"LEAK: target day {earliest.date()} is not after origin "
                f"{origin.date()} - it is inside the training period."
            )


# --------------------------------------------------------------------------- #
# The supervised training matrix
# --------------------------------------------------------------------------- #

# Shortest history that fills every feature. Derived from the window constants
# so a longer window above cannot leave this too small, which would put NaN
# features in the first rows with no error.
MIN_HISTORY = max(max(LAGS), max(ROLL_WINDOWS), 7 * max(DOW_WINDOWS))

# Bump whenever the feature definitions above change. Step 5 caches each
# series' supervised matrix keyed on this, so an edited feature is never
# served from a file built by the old definition.
FEATURE_VERSION = 2  # v2: holiday proximity (is_holiday, days_to/since_holiday)


def build_supervised(
    series: pd.DataFrame,
    horizon: int,
    use_price: bool = False,
) -> pd.DataFrame:
    """
    Every (origin, horizon) pair of one series as a supervised learning table.

    Walks the series day by day. At each origin it builds the features a real
    forecast would have and attaches the target that actually occurred. Built
    once per series and sliced per fold.

    The frame carries `origin_date` and `target_date` beside the features.
    Those two columns make leak-free training a pair of date comparisons.
      - A row may be trained on only if its `target_date` is at or before the
        fold's origin, because only then was the answer observable.
      - A row is predicted when its `origin_date` equals the fold's origin.
    """
    series = series.sort_values("date").reset_index(drop=True)
    frames = []

    for i in range(MIN_HISTORY, len(series) - horizon):
        origin = series["date"].iloc[i]
        targets = series.iloc[i + 1 : i + 1 + horizon]

        feats = build_fold_features(
            series.iloc[: i + 1],
            targets,
            origin,
            use_price=use_price,
        )
        feats["origin_date"] = origin
        feats["target_date"] = targets["date"].to_numpy()
        feats["target"] = targets["sales"].to_numpy(dtype=float)
        frames.append(feats)

    if not frames:
        # Too short for even one origin. FPP §13.6 says there is no magic
        # minimum, only that a model needs more rows than parameters; a series
        # this short gives it none. An empty matrix with the right columns
        # lets the harness skip the series. Before this the whole run died.
        frames.append(
            build_fold_features(
                series, series.iloc[0:0], series["date"].iloc[-1], use_price=use_price
            ).assign(origin_date=pd.NaT, target_date=pd.NaT, target=np.nan)
        )
    out = pd.concat(frames, ignore_index=True)
    out["id"] = series["id"].iloc[0]
    for col in POOL_ID_COLS:
        out[col] = series[col].iloc[0]
    return out


# Identity columns. Bookkeeping for a per-series model and features for a
# pooled one, which needs them to tell the series in its pool apart.
POOL_ID_COLS = ("store_id", "item_id")

NON_FEATURE_COLS = (
    "id",
    *POOL_ID_COLS,
    "date",
    "sales",
    "origin_date",
    "target_date",
    "target",
)


def supervised_feature_columns(
    pool: pd.DataFrame, pool_by: str | None = None
) -> list[str]:
    """
    The model input columns of a supervised matrix.

    `pool_by` is `Config.pool_by`. With None the model sees one series and
    the identity columns would be constant, so they are left out. With a pool
    the identity columns become features, handed over as categoricals by the
    model code.
    """
    cols = [c for c in pool.columns if c not in NON_FEATURE_COLS]
    return [*cols, *POOL_ID_COLS] if pool_by is not None else cols
