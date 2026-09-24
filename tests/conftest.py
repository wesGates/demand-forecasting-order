"""
Shared fixtures. Synthetic panels shaped like `step2_data.load_panel`'s output.

The raw M5 files are not in the repository, so every test runs on synthetic
data whose right answer is known by construction. The panel has the columns
the loader attaches (calendar flags, holiday proximity, closure), so the
harness, features and scoring code run unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.step1_problem import Config
from src.step2_data import HOLIDAY_CLIP_DAYS

STORES = {"S_BIG": ("CA", 100.0), "S_MID": ("TX", 40.0), "S_SMALL": ("WI", 15.0)}


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: fits real models; a few seconds each")


N_DAYS = 300
START = "2013-01-07"  # a Monday, like the validator's synthetic series
LAST = pd.Timestamp(START) + pd.Timedelta(days=N_DAYS - 1)


def make_panel(
    n_days: int = N_DAYS,
    start: str = START,
    stores: dict | None = None,
    holidays: tuple[str, ...] = (),
    closures: tuple[str, ...] = (),
    noise_sd: float = 3.0,
    seed: int = 0,
) -> pd.DataFrame:
    """
    One item at several stores, each a fixed weekly shape around its own
    level plus noise, with holiday-proximity columns computed the way
    `step2_data.holiday_calendar` documents them.
    """
    stores = stores or STORES
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    major = pd.to_datetime(list(holidays))
    closed = pd.to_datetime(list(closures))

    def proximity(day: pd.Timestamp) -> tuple[int, int, int]:
        ahead = [(m - day).days for m in major if m >= day]
        behind = [(day - m).days for m in major if m <= day]
        to = min(ahead + [HOLIDAY_CLIP_DAYS]) if len(ahead) else HOLIDAY_CLIP_DAYS
        since = min(behind + [HOLIDAY_CLIP_DAYS]) if len(behind) else HOLIDAY_CLIP_DAYS
        return (
            min(to, HOLIDAY_CLIP_DAYS),
            min(since, HOLIDAY_CLIP_DAYS),
            int(day in set(major)),
        )

    prox = np.array([proximity(d) for d in dates])
    frames = []
    for store, (state, level) in stores.items():
        weekly = 0.2 * level * np.sin(2 * np.pi * dates.dayofweek / 7)
        sales = level + weekly + rng.normal(0, noise_sd, n_days)
        frames.append(
            pd.DataFrame(
                {
                    "id": f"ITEM_{store}_evaluation",
                    "item_id": "ITEM",
                    "dept_id": "DEPT",
                    "cat_id": "CAT",
                    "store_id": store,
                    "state_id": state,
                    "date": dates,
                    "sales": np.maximum(sales, 0).astype("float32"),
                    "snap": (dates.day <= 10).astype("int8"),
                    "event_name_1": pd.Series([None] * n_days, dtype=object),
                    "sell_price": 2.5,
                    "is_holiday": prox[:, 2].astype("int8"),
                    "days_to_holiday": prox[:, 0].astype("int16"),
                    "days_since_holiday": prox[:, 1].astype("int16"),
                    "closure": dates.isin(closed),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def cfg(tmp_path) -> Config:
    """A small fold layout with its own cache directory."""
    return Config(
        item_ids=("ITEM",),
        n_folds=4,
        min_train_days=200,
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
    )


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return make_panel()
