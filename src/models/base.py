"""Everything a forecaster is allowed to see, and the helpers they share."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Context:
    """
    Everything a forecaster is allowed to see when producing one forecast.

    Attributes
    ----------
    history
        The series' own observed sales, ending at (and including) `origin`.
    targets
        The days being forecast: dates and known-in-advance calendar columns
        **only**. Sales for these days are deliberately absent.
    origin
        The forecast origin T. Every prediction is for T+1 .. T+horizon.
    horizon
        Number of days being forecast.
    series_id
        Which series is being forecast. Needed because a pooled `train_pool`
        holds rows for every store, and only this one's are to be predicted.
    season
        Length of the seasonal cycle in days (7 for daily retail).
    train_pool
        Supervised rows a learned model may train on, already filtered so every
        row's target was observable at `origin`. `None` for benchmarks, which
        need no training set.
    pool_by
        `Config.pool_by`, passed through. None means `train_pool` holds this
        series alone; anything else means it spans several series and the
        model is given their identity columns as features.
    seed
        Fixed, so repeat runs on identical inputs give identical output.
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
        """The observed sales history, as a float array."""
        return self.history["sales"].to_numpy(dtype=float)


Forecaster = Callable[[Context], np.ndarray]


def _flat(value: float, ctx: Context) -> np.ndarray:
    """A constant forecast repeated across the horizon."""
    return np.full(ctx.horizon, float(value))
