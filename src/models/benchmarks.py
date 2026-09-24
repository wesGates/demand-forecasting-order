"""
The benchmarks from FPP §5.2, plus two of our own. A model has to beat these
before it is worth talking about.

Six benchmarks because one misled this project early on. The seasonal naive
predicts from a single past day, so on a noisy series its error runs about
sqrt(2) higher than predicting the mean, and a model that sits near the mean
beats it with no skill at all. The mean-based ones are here to catch that.
"""

from __future__ import annotations

import numpy as np

from src.models.base import Context, Forecaster, _flat


def bench_mean(ctx: Context) -> np.ndarray:
    """
    The mean method (FPP §5.2). The average of the whole history, repeated.
    Strong on a flat series and hopeless on a trending one.
    """
    return _flat(ctx.y.mean(), ctx)


def bench_naive(ctx: Context) -> np.ndarray:
    """
    The naive method (FPP §5.2). The last value, repeated. This is the best
    you can do on a random walk.
    """
    return _flat(ctx.y[-1], ctx)


def bench_seasonal_naive(ctx: Context) -> np.ndarray:
    """
    The seasonal naive (FPP §5.2). The same weekday from the most recent
    week. This is what an orderer's default screen shows, "this day last
    week", so it is the reference benchmark for the whole study.

    Its error carries that one day's noise on top of the target's noise, which
    is why a dull model can beat it and why the mean-based benchmarks sit
    beside it.
    """
    y, s = ctx.y, ctx.season
    if len(y) < s:
        return _flat(y[-1], ctx)
    # Day h ahead maps back to the matching day of the last complete week.
    idx = [-s + ((h - 1) % s) for h in range(1, ctx.horizon + 1)]
    return y[idx]


def bench_drift(ctx: Context) -> np.ndarray:
    """
    The drift method (FPP §5.2). A straight line through the first and last
    observations, continued forward. The naive forecast with a trend allowed.
    """
    y = ctx.y
    if len(y) < 2:
        return _flat(y[-1], ctx)
    slope = (y[-1] - y[0]) / (len(y) - 1)
    return y[-1] + slope * np.arange(1, ctx.horizon + 1)


def bench_moving_average(ctx: Context, window: int = 28) -> np.ndarray:
    """
    A flat forecast at the mean of the last `window` days. Not one of FPP's
    four. It follows the recent level and is the hardest benchmark here. An
    earlier version of this project claimed a 75% win rate against the
    seasonal naive that fell to 42% against this one.
    """
    return _flat(ctx.y[-window:].mean(), ctx)


def bench_seasonal_naive_364(ctx: Context) -> np.ndarray:
    """
    The seasonal naive with a one-year period. The same weekday 52 weeks ago,
    which is what an experienced orderer looks at before a holiday. Lag 364
    keeps the weekday aligned (FPP §13.1 covers annual periods in daily data).
    Falls back to the weekly version until a year of history exists.
    """
    y = ctx.y
    lag = 52 * ctx.season
    if len(y) < lag:
        return bench_seasonal_naive(ctx)
    idx = [-lag + (h - 1) for h in range(1, ctx.horizon + 1)]
    return y[idx]


BENCHMARKS: dict[str, Forecaster] = {
    "mean": bench_mean,
    "naive": bench_naive,
    "seasonal_naive": bench_seasonal_naive,
    "seasonal_naive_364": bench_seasonal_naive_364,
    "drift": bench_drift,
    "moving_average_28": bench_moving_average,
}
