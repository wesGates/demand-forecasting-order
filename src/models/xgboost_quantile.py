"""
Gradient-boosted trees trained on the quantile objective.

The point model minimises squared error, so it estimates the conditional
mean, and an order built from it under-covers demand about half the time.
This model minimises pinball loss at several service levels at once
(`reg:quantileerror` with a vector of quantile levels), so each of its
outputs is the τ-quantile of daily demand directly - the number an order
needs, without a calibration step afterwards.

One fit per store and origin produces every quantile; the registry exposes
one forecaster per τ (`xgboost_q30`, ..., `xgboost_q90`) that all read the
same fitted model, memoised per run. Features, early stopping and clipping
are identical to the point model: the two differ in the objective only,
which is what makes the comparison a comparison.
"""

from __future__ import annotations

import numpy as np

from src.models.base import Context, Forecaster
from src.models.xgboost_model import XGB_PARAMS, design

# The service levels fitted, matching the ones the ordering analysis reports.
TAUS = (0.3, 0.5, 0.7, 0.9)


def method_name(tau: float) -> str:
    return f"xgboost_q{int(round(tau * 100)):02d}"


# Predictions for every τ, per (series, origin), for the current run. The
# harness clears this at the start of every run through `reset()`.
_predicted: dict[tuple, np.ndarray] = {}


def reset() -> None:
    _predicted.clear()


def _fit_all(ctx: Context) -> np.ndarray:
    """(horizon, len(TAUS)) array of quantile forecasts, fitted once per fold."""
    from xgboost import XGBRegressor

    key = (ctx.series_id, ctx.origin)
    if key in _predicted:
        return _predicted[key]

    parts = design(ctx)
    if parts is None:
        out = np.full((ctx.horizon, len(TAUS)), np.nan)
        _predicted[key] = out
        return out
    x_train, y_train, x_pred, inner_fit = parts
    params = {
        **XGB_PARAMS,
        "objective": "reg:quantileerror",
        "quantile_alpha": np.array(TAUS),
        "random_state": ctx.seed,
    }

    if inner_fit.sum() and (~inner_fit).sum():
        model = XGBRegressor(**params, enable_categorical=True, early_stopping_rounds=50)
        model.fit(
            x_train[inner_fit],
            y_train[inner_fit],
            eval_set=[(x_train[~inner_fit], y_train[~inner_fit])],
            verbose=False,
        )
    else:
        model = XGBRegressor(**{**params, "n_estimators": 150}, enable_categorical=True)
        model.fit(x_train, y_train)

    q = np.asarray(model.predict(x_pred), dtype=float).reshape(len(x_pred), len(TAUS))
    # Quantiles fitted separately can cross; a higher service level must never
    # order less than a lower one, so they are sorted per day.
    q = np.sort(np.clip(q, 0, None), axis=1)
    _predicted[key] = q
    return q


def _forecaster(tau: float) -> Forecaster:
    col = TAUS.index(tau)

    def fit_predict(ctx: Context) -> np.ndarray:
        return _fit_all(ctx)[:, col]

    fit_predict.__name__ = f"fit_predict_{method_name(tau)}"
    fit_predict.__doc__ = (
        f"The τ = {tau} quantile of daily demand from the quantile-objective XGBoost."
    )
    return fit_predict


QUANTILE_MODELS: dict[str, Forecaster] = {method_name(t): _forecaster(t) for t in TAUS}
