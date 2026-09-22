"""Holt-Winters exponential smoothing with weekly seasonality (FPP Ch. 8)."""

from __future__ import annotations

import warnings

import numpy as np

from src.models.base import Context, _flat


def fit_predict_ets(ctx: Context) -> np.ndarray:
    """
    Holt-Winters exponential smoothing with weekly seasonality (FPP Ch. 8).

    A real classical contender rather than a benchmark: a weighted average of
    the past where the weights decay exponentially, extended to carry a trend
    and a repeating weekly shape. On a smooth, strongly seasonal series it is
    often very hard to beat, which is exactly the case this study is about.

    Fitted on the most recent two years only. ETS weights recent observations
    most heavily anyway, and the full five years both slows the optimiser and
    drags the fitted seasonal shape toward a level the series left behind -
    step 3 showed a clear multi-year downward drift.

    If the optimiser fails, the forecast falls back to the 28-day mean and a
    warning names the series and origin. The fallback keeps a single bad fold
    from aborting a whole run; the warning keeps it from hiding - a silent
    fallback would let "ETS diverged every time" masquerade as "ETS is weak".
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    y = ctx.y[-730:]
    if len(y) < 2 * ctx.season:
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
        return _flat(ctx.y[-28:].mean(), ctx)
