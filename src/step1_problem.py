"""
Step 1 - Problem definition (FPP §1.6, step 1).

FPP §1.3 says to pin the forecasting task down before touching data: what is
forecast, at what grain, how far ahead, how success is judged. This module is
that decision in one object. Changing the horizon, the fold count or the data
subset means editing one line here. Each field has a comment on what it does
and which FPP section applies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Repo root, taken from this file's location. Relative paths in a Config are
# resolved against it, so a Config means the same thing from the project root
# or from inside notebooks/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The study item. One fast-moving staple sold at all ten stores, picked by the
# availability screen in step 3 (longest zero run 10 days over 1,885 days).
# Volumes run 15-101 units/day across the stores, and that spread is what the
# model comparison runs along.
STUDY_ITEMS = ("FOODS_3_586",)

# Three stores across the volume range, eight weekly folds. For trying a model
# change before paying for a full run. A point-model change takes about 2 s
# here, ARIMA about 30 s. Use as `Config(**DEV)`.
DEV = dict(item_ids=STUDY_ITEMS, store_ids=("CA_4", "TX_2", "WI_2"), n_folds=8)  # sorted, as Config keeps them

# Columns a model is allowed to pool over. Anything else is a typo.
POOL_SCOPES = (None, "item_id", "dept_id", "cat_id", "store_id", "state_id")

# Where the RMSSE denominator comes from.
#   "pre_holdout" - one denominator per series, from all training data before
#                   the first scored day. Shared by every fold and method.
#   "per_fold"    - recomputed from each fold's own history. Closer to a single
#                   train/test split, but the scale drifts between folds.
RMSSE_SCALE_WINDOWS = ("pre_holdout", "per_fold")


@dataclass(frozen=True)
class Config:
    """
    The complete definition of one forecasting experiment.

    Frozen on purpose. A config that changes halfway through a notebook
    session cannot be trusted when the result is reported. Each field has its
    explanation in the comment beside it.
    """

    # ======================================================================
    # What we forecast
    # ======================================================================

    # Days ahead. One forecast for each h = 1..horizon from a single origin T,
    # written ŷ_{T+h|T} in FPP §1.7. Daily because inventory is consumed day
    # by day, and a weekly total loses the weekday pattern that decides when
    # a store runs out.
    horizon: int = 7

    # Days scored per fold. Defaults to `horizon` and may never exceed it (see
    # the safeguard in __post_init__). This one comes from a bug the project
    # already had, not from the book.
    test_window: int | None = None

    # Length of the seasonal cycle in days. 7 for daily retail. Used by the
    # seasonal naive benchmark (FPP §5.2), by ETS as its seasonal period
    # (§8.3), and as the default RMSSE scaling lag (§5.8).
    season: int = 7

    # ======================================================================
    # How we validate
    # ======================================================================

    # Number of walk-forward folds, FPP §5.10's rolling forecasting origin.
    # One origin per fold, `fold_step` days apart. 52 folds of 7 days is one
    # full year, about 20% of the five-year history, and the shortest layout
    # that scores every season once. The book gives no fold count.
    n_folds: int = 52

    # Days between fold origins.
    #   7  - origins tile the year with no overlap. Every origin is the same
    #        weekday (a Sunday here), so every claim is a Sunday-order claim
    #        and the horizon plot mixes horizon with weekday.
    #   1  - every day is an origin, as FPP's own example does. Origins rotate
    #        through the week. Windows overlap, so adjacent folds share six of
    #        their seven days and 358 folds are not 358 independent samples.
    #   8  - origins rotate through the week with one-day gaps. Not used.
    # Step 1 is the reported layout since item 2. Step 7 is for quick checks.
    fold_step: int = 7

    # A series needs at least this many days of history at a fold's origin or
    # that fold is skipped for it. 365 keeps one annual cycle in view. Project
    # decision. (The feature module separately needs `features.MIN_HISTORY`
    # days to fill every lag, a much smaller number.)
    min_train_days: int = 365

    # ======================================================================
    # How we score
    # ======================================================================

    # RMSSE divides the forecast error by the error of a naive forecast on the
    # training data, so it is comparable across stores of different size (FPP
    # §5.8). This is the lag of that naive forecast. The book says lag 1 for
    # non-seasonal data and lag m for seasonal data, and this data is weekly
    # seasonal. None means "use `season`". The M5 competition used lag 1; set
    # 1 to compare against published M5 numbers.
    rmsse_scale_lag: int | None = None

    # Which training data the RMSSE denominator comes from. FPP §5.8 says "the
    # training set" and §5.10 does not say what that means with rolling
    # origins, so this is a project decision. See RMSSE_SCALE_WINDOWS above.
    rmsse_scale_window: str = "pre_holdout"

    # ======================================================================
    # Which data
    # ======================================================================

    # Filters applied before the wide-to-long reshape. An empty tuple or None
    # means no filter on that column. Always pass something in interactive
    # work; the unfiltered panel is about 59 million rows.
    item_ids: tuple[str, ...] = ()
    store_ids: tuple[str, ...] = ()
    dept_id: str | None = None
    cat_id: str | None = None

    # ======================================================================
    # How we model
    # ======================================================================

    # Which rows a learned model may train on.
    #   None       -> one model per store-item, trained on that series alone
    #   "item_id"  -> one model per item, trained on all of its stores at once
    # Wider scopes use the same machinery. The model sees every series in the
    # pool and gets `store_id` and `item_id` as categorical features to tell
    # them apart. This comes from the M5 competition, where every winning
    # method pooled, and the book does not cover it. Only None and "item_id"
    # have been run so far.
    pool_by: str | None = None

    # Whether `sell_price` is offered to the model. Price is known a week ahead
    # in this dataset, so it is a legitimate predictor in FPP's sense (§7,
    # predictors known in advance). Off by default. For the study item it
    # takes three values in five years, on the same two dates at every store,
    # and never changes inside a forecast window, so it acts as a clock. Tested
    # on 2026-09-21 and slightly worse with it on. An item whose price moves
    # may want it back.
    use_price: bool = False


    # Fixed so repeat runs on the same inputs give the same output. The
    # validator checks this.
    seed: int = 0

    # ======================================================================
    # Where things live (resolved against PROJECT_ROOT)
    # ======================================================================

    data_dir: Path = Path("data")
    cache_dir: Path = Path("cache")

    def __post_init__(self) -> None:
        # Sorted, so ("A", "B") and ("B", "A") are one config and one cache
        # key.
        object.__setattr__(self, "item_ids", tuple(sorted(self.item_ids)))
        object.__setattr__(self, "store_ids", tuple(sorted(self.store_ids)))
        if self.test_window is None:
            object.__setattr__(self, "test_window", self.horizon)
        if self.rmsse_scale_lag is None:
            object.__setattr__(self, "rmsse_scale_lag", self.season)

        # --- the safeguard --------------------------------------------------
        # Features are lagged by the horizon. A scored window longer than the
        # horizon would let the lags for its later days point at dates inside
        # the window, and the model would read the actuals it is supposed to
        # predict. Known mistake: this produced a fabricated result once, so
        # it is asserted here.
        if self.test_window > self.horizon:
            raise ValueError(
                f"test_window ({self.test_window}) must not exceed horizon "
                f"({self.horizon}). A longer scored window lets lag features "
                f"reach into the period being forecast. Recover sample size "
                f"with more folds (n_folds), never with a longer window."
            )
        if self.horizon < 1 or self.n_folds < 1:
            raise ValueError("horizon and n_folds must both be >= 1.")
        if self.fold_step < 1:
            raise ValueError("fold_step must be >= 1 day.")
        if self.rmsse_scale_lag < 1:
            raise ValueError("rmsse_scale_lag must be >= 1.")

        # Catch a mistyped option here, before it fails deep inside a model.
        if self.pool_by not in POOL_SCOPES:
            raise ValueError(
                f"pool_by={self.pool_by!r} is not a poolable column. "
                f"Use one of: {POOL_SCOPES}"
            )
        if self.rmsse_scale_window not in RMSSE_SCALE_WINDOWS:
            raise ValueError(
                f"rmsse_scale_window={self.rmsse_scale_window!r} is not an option. "
                f"Use one of: {RMSSE_SCALE_WINDOWS}"
            )

        # Relative paths are anchored to the repo root, so the working
        # directory does not change what a Config means. Absolute paths pass
        # through.
        for p in ("data_dir", "cache_dir"):
            value = Path(getattr(self, p))
            if not value.is_absolute():
                value = PROJECT_ROOT / value
            object.__setattr__(self, p, value)

    # ---------------------------------------------------------------------- #

    @property
    def subset(self) -> dict[str, object]:
        """The data filter as a plain dict, for cache keys and captions."""
        return {
            "item_ids": tuple(self.item_ids),
            "store_ids": tuple(self.store_ids),
            "dept_id": self.dept_id,
            "cat_id": self.cat_id,
        }

    def holdout_start(self, last_date):
        """
        The first date that will ever be scored under this fold layout.

        The last fold ends on the final day of data. Earlier folds step back
        `fold_step` days at a time, so the earliest scored day is
        `(n_folds - 1) * fold_step + test_window` days from the end.

        Step 3 uses this so no class label is computed from a scored day, and
        step 5 uses it to lay out the folds. Computed in one place so the two
        cannot drift apart.
        """
        return last_date - pd.Timedelta(days=self.test_days_total - 1)

    def fold_origins(self, last_date) -> list:
        """
        The forecast origin of every walk-forward fold, earliest first.

        Fold k scores the `test_window` days that begin `k * fold_step` days
        after `holdout_start`. Its origin is the day before the first of them.
        With `fold_step == test_window` the folds tile the held-out period
        exactly, with a smaller step they overlap, and the last fold always
        ends on the final day of data.
        """
        start = self.holdout_start(last_date)
        return [
            start + pd.Timedelta(days=k * self.fold_step - 1) for k in range(self.n_folds)
        ]

    @property
    def test_days_total(self) -> int:
        """Days from the first scored day to the last, inclusive."""
        return (self.n_folds - 1) * self.fold_step + self.test_window

    def describe(self) -> str:
        """A readable summary for the top of a notebook."""
        if self.pool_by is None:
            scope = "per store-item"
        else:
            scope = f"pooled by {self.pool_by}"

        filters = [f"{k}={v}" for k, v in self.subset.items() if v not in ((), None)]
        where = ", ".join(filters) if filters else "whole panel"

        return (
            f"Forecast daily unit sales, h=1..{self.horizon} from origin T.\n"
            f"  subset      : {where}\n"
            f"  validation  : {self.n_folds} walk-forward folds, origins {self.fold_step} day(s) apart, "
            f"x {self.test_window}-day window ({self.test_days_total} days scored)\n"
            f"  training    : {scope}, min {self.min_train_days} days, "
            f"price {'on' if self.use_price else 'off'}\n"
            f"  scoring     : RMSSE scaled by lag-{self.rmsse_scale_lag} naive error, "
            f"{self.rmsse_scale_window} window\n"
            f"  seed        : {self.seed}"
        )


if __name__ == "__main__":
    print(Config(item_ids=STUDY_ITEMS).describe(), "\n")
    for bad in (
        dict(test_window=28),
        dict(pool_by="item"),
        dict(rmsse_scale_window="all"),
    ):
        try:
            Config(**bad)
        except ValueError as e:
            print(f"rejected {bad}:\n  {e}\n")
