"""
Gradient-boosted trees trained on the quantile objective.

The point model minimises squared error and so estimates the mean. An order
placed at the mean runs out about half the time. This model minimises
pinball loss at several service levels at once (`reg:quantileerror` with a
vector of quantile levels), so each output is the τ-quantile of daily
demand, the number an order actually needs.

Item 1 compared it with the cheaper route of adding the point model's own
error quantiles to its forecast. The two matched within 0.002 relative
pinball at every τ, and this one cost 26 times as much to fit. It stays in
the registry as the comparator for that result.

One fit per store and origin gives every quantile. The registry exposes one
forecaster per τ (`xgboost_q30` to `xgboost_q90`) and they all read the
same fit. Features, early stopping and clipping are the point model's.
"""

from __future__ import annotations

import numpy as np

from src.models.base import Context, Forecaster
from src.models.xgboost_model import XGB_PARAMS, design

# The service levels fitted. Same four the ordering analysis reports.
TAUS = (0.3, 0.5, 0.7, 0.9)


def method_name(tau: float) -> str:
    return f"xgboost_q{int(round(tau * 100)):02d}"


# Predictions for every τ, keyed by (series, origin). One fit serves the four
# forecasters, so the first one to ask pays for it. The harness clears this
# through `reset()` at the start of every run and every task.
_predicted: dict[tuple, np.ndarray] = {}


def reset() -> None:
    _predicted.clear()


def _fit_all(ctx: Context) -> np.ndarray:
    """A (horizon, len(TAUS)) array of quantile forecasts, fitted once per fold."""
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
    # Quantiles fitted seperately can cross. A higher service level must not
    # order less than a lower one, so each day's four values are sorted.
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
