"""What a forecaster is allowed to see, plus the small helpers every model shares."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Context:
    """
    Everything a forecaster gets for one forecast. Nothing from after the
    origin is in here, so a model cannot read the answer by accident.

    history
        The series' own sales up to and including `origin`.
    targets
        The days being forecast. Dates and calendar columns only, no sales.
    origin
        The day T the forecast is made from. Predictions cover T+1 to T+horizon.
    horizon
        How many days ahead.
    series_id
        Which series this is. A pooled `train_pool` holds every store, and
        only this one gets predicted.
    season
        Length of the seasonal cycle in days. 7 for daily retail.
    train_pool
        Supervised rows a learned model may train on. None for benchmarks.
        The model filters these itself to rows whose target was known at the
        origin.
    pool_by
        `Config.pool_by`, passed through. None means the pool is this series
        alone. Anything else means several series share it and the model gets
        their identity columns as features.
    seed
        Fixed so the same inputs give the same forecasts every time.
    """

    history: pd.DataFrame
    targets: pd.DataFrame
    origin: pd.Timestamp
    horizon: int
    series_id: str

    season: int = 7
    train_pool: pd.DataFrame | None = None
    pool_by: str | None = None
    seed: int = 0

    @property
    def y(self) -> np.ndarray:
        """The sales history as a float array."""
        return self.history["sales"].to_numpy(dtype=float)


Forecaster = Callable[[Context], np.ndarray]

# A forecaster that gives up on a fold (a failed fit, too little history)
# appends a reason here. The harness clears the list before each call, checks
# it after, and flags the row. Before this existed a failed ARIMA fit returned
# the 28-day mean and got scored as ARIMA, which made ARIMA look better than
# it was.
fallbacks: list[str] = []


def note_fallback(reason: str) -> None:
    fallbacks.append(reason)


def _flat(value: float, ctx: Context) -> np.ndarray:
    """One value repeated for every day of the horizon."""
    return np.full(ctx.horizon, float(value))
