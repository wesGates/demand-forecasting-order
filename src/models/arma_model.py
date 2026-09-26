"""
Two weaker relatives of the ARIMA model, as comparison baselines.

`arima_plain` is the seasonal ARIMA with its holiday and SNAP regressors
removed, so the gap to `arima` is the measured value of those inputs.
`arma` is a non-seasonal ARMA with a constant, no differencing and no
regressors, the kind of baseline an earlier study used on weekly totals.
On daily data with a 50% weekday swing it is expected to do badly, and that
is the point of having it.

Orders are chosen by AICc on a fixed window that ends the day before the
test year, so the choice never sees a scored day and does not depend on
which worker asks first. Known shortcut: the cutoff is a constant here
rather than read from the Config, because the forecaster Context does not
carry it. It matches both layouts of this study.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.models.arima_model import ARIMA_FIT_DAYS
from src.models.base import Context, _flat, note_fallback

SELECT_BEFORE = pd.Timestamp("2015-05-25")  # the test year starts here on both layouts

# (p, q, P, Q). The seasonal variant keeps ARIMA's grid and its one seasonal
# difference; the ARMA grid has no seasonal part.
GRID_SEASONAL = [(p, q, P, Q) for p in (0, 1, 2) for q in (0, 1, 2) for P in (0, 1) for Q in (0, 1)]
GRID_ARMA = [(p, q, 0, 0) for p in (0, 1, 2, 3) for q in (0, 1, 2, 3)]

_orders: dict[tuple[str, str], tuple] = {}


def reset() -> None:
    _orders.clear()


def _aicc(fit, n: int) -> float:
    k = len(fit.params)
    return fit.aic + (2 * k * (k + 1)) / max(n - k - 1, 1)


def _select(y: np.ndarray, grid, seasonal: bool, season: int):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    best = None
    for p, q, P, Q in grid:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = SARIMAX(
                    y,
                    order=(p, 0, q),
                    seasonal_order=(P, 1, Q, season) if seasonal else (0, 0, 0, 0),
                    trend="n" if seasonal else "c",  # the ARMA needs a mean; the seasonal difference removes it
                ).fit(disp=False, maxiter=200)
            score = _aicc(fit, len(y))
            if best is None or score < best[2]:
                best = ((p, 0, q), (P, 1, Q, season) if seasonal else (0, 0, 0, 0), score)
        except Exception:
            continue
    if best is None:
        raise RuntimeError("no order in the grid could be fitted")
    return best


def _fit_predict(ctx: Context, name: str, seasonal: bool) -> np.ndarray:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    hist = ctx.history.iloc[-ARIMA_FIT_DAYS:]
    y = hist["sales"].to_numpy(dtype=float)
    if len(y) < 4 * ctx.season:
        note_fallback(f"{name}: 28-day mean")
        return _flat(ctx.y[-28:].mean(), ctx)
    try:
        key = (name, ctx.series_id)
        if key not in _orders:
            before = ctx.history[ctx.history["date"] < SELECT_BEFORE].iloc[-ARIMA_FIT_DAYS:]
            _orders[key] = _select(before["sales"].to_numpy(dtype=float), GRID_SEASONAL if seasonal else GRID_ARMA, seasonal, ctx.season)
        order, seasonal_order, _ = _orders[key]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = SARIMAX(y, order=order, seasonal_order=seasonal_order, trend="n" if seasonal else "c").fit(disp=False, maxiter=200)
            fc = np.asarray(fit.forecast(steps=ctx.horizon), dtype=float)
        # A fit that converged to nonsense passes through as a forecast unless
        # it is caught here. One ARMA fold on the fast item forecast 17 million.
        if not np.all(np.isfinite(fc)) or fc.max() > 10 * max(y.max(), 1.0):
            note_fallback(f"{name}: diverged fit, 28-day mean")
            return _flat(ctx.y[-28:].mean(), ctx)
        return np.clip(fc, 0, None)
    except Exception as err:
        warnings.warn(f"{name} failed for {ctx.series_id} at origin {ctx.origin.date()} ({type(err).__name__}: {err}); using the 28-day mean instead.", stacklevel=2)
        note_fallback(f"{name}: 28-day mean")
        return _flat(ctx.y[-28:].mean(), ctx)


def fit_predict_arima_plain(ctx: Context) -> np.ndarray:
    """Seasonal ARIMA without the holiday and SNAP regressors."""
    return _fit_predict(ctx, "arima_plain", seasonal=True)


def fit_predict_arma(ctx: Context) -> np.ndarray:
    """Non-seasonal ARMA with a constant. No differencing, no regressors."""
    return _fit_predict(ctx, "arma", seasonal=False)
