"""Seasonal ARIMA with calendar regressors (FPP ch. 9; ch. 10 for the regressors)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.models.base import Context, _flat, note_fallback

# The regressors ARIMA gets. All three are known in advance for any target
# date, which makes them fair predictors (FPP §10.1). `pre_holiday` is the
# two-day run-up the event-effect table showed for this item.
ARIMA_EXOG = ("is_holiday", "pre_holiday", "snap")

# The differencing is fixed. No ordinary difference, because the series is
# level-stationary over a two-year window, and one seasonal difference at
# lag 7. FPP §9.7 says information criteria cannot compare models with
# different differencing, so the grid searches only the AR and MA orders.
ARIMA_D, ARIMA_SEASONAL_D = 0, 1
ARIMA_GRID = [
    (p, q, P, Q) for p in (0, 1, 2) for q in (0, 1, 2) for P in (0, 1) for Q in (0, 1)
]
ARIMA_FIT_DAYS = 730

# One chosen order per series, picked on the series' first scored window and
# held for the rest of the run. The grid search takes about 24 s per series,
# so doing it once per fold would have turned a 10 min run into hours. The
# harness picks the orders before the folds start and clears this dict at
# the start of every run, so a second layout in the same process starts
# fresh.
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
    """The lowest-AICc order in ARIMA_GRID. Returns (order, seasonal_order, aicc)."""
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
    Seasonal ARIMA with regressors (FPP ch. 9; ch. 10 for the regressors).

    The classical model that can be told about the calendar. It gets the same
    known-in-advance flags XGBoost gets (holiday, pre-holiday, SNAP), so the
    two are compared on modelling and not on who saw the calendar.

    Fitted on the last two years, like ETS. The order is chosen once per
    series by AICc and held across folds; see `arima_orders` for why.

    A failed fit returns the 28-day mean, warns, and notes the fallback.
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
