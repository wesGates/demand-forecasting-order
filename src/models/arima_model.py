"""Seasonal ARIMA with calendar regressors (FPP Ch. 9, and Ch. 10 for the regressors)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.models.base import Context, _flat, note_fallback

# Regressors handed to ARIMA. All are known in advance for any target date, so
# they are legitimate predictors in FPP's sense (§10.1). `pre_holiday` is the
# two-day run-up the event-effect table shows for this item.
ARIMA_EXOG = ("is_holiday", "pre_holiday", "snap")

# The differencing is fixed, not searched: no ordinary difference (the series
# is level-stationary over two years) and one seasonal difference at lag 7.
# FPP §9.7 is explicit that information criteria cannot compare models with
# different orders of differencing, so only the AR and MA orders are chosen.
ARIMA_D, ARIMA_SEASONAL_D = 0, 1
ARIMA_GRID = [
    (p, q, P, Q) for p in (0, 1, 2) for q in (0, 1, 2) for P in (0, 1) for Q in (0, 1)
]
ARIMA_FIT_DAYS = 730

# Chosen orders, one per series, filled on the first fold each series is
# forecast in a run and reused for the rest of that run. The harness walks
# folds oldest first, so the selection is made on the earliest training
# window and never sees a scored day - and `run_walk_forward` clears this at
# the start of every run, so a second layout in the same process selects
# afresh on its own first window.
arima_orders: dict[
    str, tuple[tuple[int, int, int], tuple[int, int, int, int], float]
] = {}


def _arima_exog(frame: pd.DataFrame) -> np.ndarray:
    out = pd.DataFrame(index=frame.index)
    out["is_holiday"] = frame["is_holiday"].astype(float)
    out["pre_holiday"] = frame["days_to_holiday"].between(1, 2).astype(float)
    out["snap"] = frame["snap"].astype(float)
    return out[list(ARIMA_EXOG)].to_numpy(dtype=float)


def _select_arima_order(y: np.ndarray, exog: np.ndarray, season: int):
    """AICc over ARIMA_GRID at fixed differencing. Returns (order, seasonal_order, aicc)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    best = None
    for p, q, P, Q in ARIMA_GRID:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = SARIMAX(
                    y,
                    exog=exog,
                    order=(p, ARIMA_D, q),
                    seasonal_order=(P, ARIMA_SEASONAL_D, Q, season),
                ).fit(disp=False, maxiter=200)
            aicc = float(fit.aicc)
        except Exception:
            continue
        if np.isfinite(aicc) and (best is None or aicc < best[2]):
            best = ((p, ARIMA_D, q), (P, ARIMA_SEASONAL_D, Q, season), aicc)
    if best is None:
        raise RuntimeError("no ARIMA order in the grid could be fitted")
    return best


def fit_predict_arima(ctx: Context) -> np.ndarray:
    """
    Seasonal ARIMA with regressors (FPP Ch. 9, and Ch. 10 for the regressors).

    The classical model that *can* be told about the calendar. It is given the
    same known-in-advance information XGBoost gets - a holiday flag, a
    pre-holiday flag and the SNAP flag - so the comparison between the two is
    about the modelling, not about who was allowed to see the calendar.

    Fitted on the most recent two years, like ETS and for the same reason. The
    order is selected once per series by AICc on that series' first training
    window and then held fixed across folds: re-selecting on every fold would
    multiply the run time by the grid size and make fold-to-fold differences
    partly about which order happened to win.

    Fallback and warning follow ETS's pattern: a failed fit gives the 28-day
    mean, and says so.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    hist = ctx.history.iloc[-ARIMA_FIT_DAYS:]
    y = hist["sales"].to_numpy(dtype=float)
    if len(y) < 4 * ctx.season:
        note_fallback("arima: 28-day mean")
        return _flat(ctx.y[-28:].mean(), ctx)
    x_hist, x_future = _arima_exog(hist), _arima_exog(ctx.targets)

    try:
        if ctx.series_id not in arima_orders:
            arima_orders[ctx.series_id] = _select_arima_order(y, x_hist, ctx.season)
        order, seasonal_order, _ = arima_orders[ctx.series_id]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = SARIMAX(y, exog=x_hist, order=order, seasonal_order=seasonal_order).fit(
                disp=False, maxiter=200
            )
            fc = fit.forecast(steps=ctx.horizon, exog=x_future[: ctx.horizon])
        return np.clip(np.asarray(fc, dtype=float), 0, None)
    except Exception as err:
        warnings.warn(
            f"ARIMA failed for {ctx.series_id} at origin {ctx.origin.date()} "
            f"({type(err).__name__}: {err}); using the 28-day mean instead.",
            stacklevel=2,
        )
        note_fallback("arima: 28-day mean")
        return _flat(ctx.y[-28:].mean(), ctx)
