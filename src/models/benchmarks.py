"""
Benchmarks - FPP Ch. 5. What every model must beat to be interesting.

All six are here, not one. A single benchmark has already misled this project
once: seasonal naive predicts one specific past day, so on a noisy series its
error runs about sqrt(2) worse than simply predicting the mean, and a model
that predicts near the mean "wins" without any skill at all. The mean-based
methods are the guard against that.
"""

from __future__ import annotations

import numpy as np

from src.models.base import Context, Forecaster, _flat


def bench_mean(ctx: Context) -> np.ndarray:
    """
    Mean method: the average of all history.

    FPP §5.2. Unbeatable-looking on a stationary series and hopeless on a
    trending one, which is precisely why it belongs alongside the others.
    """
    return _flat(ctx.y.mean(), ctx)


def bench_naive(ctx: Context) -> np.ndarray:
    """
    Naive method: the last observed value, repeated.

    FPP §5.2, `ŷ_{T+h|T} = y_T`. Optimal for a random walk, and the benchmark
    any forecast of a genuinely unpredictable series should be measured against.
    """
    return _flat(ctx.y[-1], ctx)


def bench_seasonal_naive(ctx: Context) -> np.ndarray:
    """
    Seasonal naive: the value from the same weekday in the most recent week.

    FPP §5.2. Captures the weekly pattern for free and is what a person with no
    tools would do.

    Its weakness is the reason this project reports six benchmarks: it stakes
    everything on one specific past day, so its error carries that day's noise
    *plus* the target's. On a noisy series that is roughly sqrt(2) worse than
    predicting the mean, and a model can beat it by being sensibly dull.
    """
    y, s = ctx.y, ctx.season
    if len(y) < s:
        return _flat(y[-1], ctx)
    # h = 1..horizon maps back to the matching day of the last complete cycle.
    idx = [-s + ((h - 1) % s) for h in range(1, ctx.horizon + 1)]
    return y[idx]


def bench_drift(ctx: Context) -> np.ndarray:
    """
    Drift method: the last value, extrapolated along the average historical
    slope.

    FPP §5.2. Equivalent to drawing a straight line through the first and last
    observations and continuing it - a naive forecast that is allowed a trend.
    """
    y = ctx.y
    if len(y) < 2:
        return _flat(y[-1], ctx)
    slope = (y[-1] - y[0]) / (len(y) - 1)
    return y[-1] + slope * np.arange(1, ctx.horizon + 1)


def bench_moving_average(ctx: Context, window: int = 28) -> np.ndarray:
    """
    Flat forecast at the mean of the last `window` days.

    Not one of FPP's four, and included because it is the one that matters. It
    tracks the recent level rather than the whole history, and on this kind of
    data it is a genuinely hard benchmark to beat - a previous version of this
    project found a supposed 75% win rate collapse to 42% when measured against
    it instead of seasonal naive.
    """
    return _flat(ctx.y[-window:].mean(), ctx)


def bench_seasonal_naive_364(ctx: Context) -> np.ndarray:
    """
    Seasonal naive with a one-year period: the same weekday, 52 weeks ago.

    The weekly seasonal naive is what an orderer's default screen shows - this
    day last week. Before a holiday an experienced orderer switches to *this
    day last year*, and that is what this benchmark encodes. Lag 364 rather
    than 365 keeps the weekday aligned (FPP §13.1 on annual periods in daily
    data). Falls back to the weekly version when a year of history is not yet
    available, so the two benchmarks are identical on short series.

    Included so that the holiday-week comparison is against what a good orderer
    actually does, not only against the default screen.
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
