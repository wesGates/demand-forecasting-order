"""Holt-Winters exponential smoothing with a weekly season (FPP ch. 8)."""

from __future__ import annotations

import warnings

import numpy as np

from src.models.base import Context, _flat, note_fallback


def fit_predict_ets(ctx: Context) -> np.ndarray:
    """
    Holt-Winters exponential smoothing with a weekly season (FPP ch. 8).

    A weighted average of the past with the weights decaying exponentially,
    plus a trend and a repeating weekly shape. Hard to beat on a smooth,
    seasonal series, which is the kind this study started with.

    Fitted on the last two years only. The full five years slowed the
    optimiser and pulled the seasonal shape toward a level the series had
    left behind (step 3 found a multi-year downward drift), so the fit
    window was cut to 730 days.

    A failed fit returns the 28-day mean, warns with the series and origin,
    and notes the fallback so the harness can count it. Without the note a
    run where ETS diverged on every fold would look like ETS being weak.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    y = ctx.y[-730:]
    if len(y) < 2 * ctx.season:
        note_fallback("ets: 28-day mean")
        return _flat(ctx.y[-28:].mean(), ctx)

    try:
        fit = ExponentialSmoothing(
            y,
            trend="add",
            seasonal="add",
            seasonal_periods=ctx.season,
            initialization_method="estimated",
        ).fit()
        return np.clip(np.asarray(fit.forecast(ctx.horizon), dtype=float), 0, None)
    except Exception as err:
        warnings.warn(
            f"ETS failed for {ctx.series_id} at origin {ctx.origin.date()} "
            f"({type(err).__name__}: {err}); using the 28-day mean instead.",
            stacklevel=2,
        )
        note_fallback("ets: 28-day mean")
        return _flat(ctx.y[-28:].mean(), ctx)
