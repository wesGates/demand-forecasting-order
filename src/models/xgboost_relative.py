"""
Gradient-boosted trees on a level-relative target.

Item 4 showed why a tree model loses on a declining item: it predicts from
leaves grown on the levels it trained on and cannot extrapolate below them,
so as sales fall its forecast stays where the history was. Exponential
smoothing tracks the level by construction.

This variant keeps every feature and every setting of the point model and
changes only what the trees are asked to predict: the target is the day's
sales *minus the 28-day mean at the origin* (a feature the row already
carries), and the forecast is the prediction plus that same mean. The
trees then learn the weekly and holiday shape around the recent level, and
the level itself is carried by the most recent four weeks - which is what a
moving average does, and what the trees could not.

Per-store and pooled variants follow the point model exactly, including the
one-fit-per-origin memo for the pooled case.
"""

from __future__ import annotations

import numpy as np

from src.models.base import Context
from src.models.xgboost_model import XGB_PARAMS, _fit, design

LEVEL_FEATURE = "roll_mean_28"

_pooled_models: dict[tuple, object] = {}


def reset() -> None:
    _pooled_models.clear()


def fit_predict_xgboost_relative(ctx: Context, **overrides) -> np.ndarray:
    """The point model's trees, trained on sales minus the origin's 28-day mean."""
    parts = design(ctx)
    if parts is None:
        return np.full(ctx.horizon, np.nan)
    x_train, y_train, x_pred, inner_fit = parts
    level_train = x_train[LEVEL_FEATURE].to_numpy(dtype=float)
    level_pred = x_pred[LEVEL_FEATURE].to_numpy(dtype=float)
    params = {**XGB_PARAMS, **overrides, "random_state": ctx.seed}

    if ctx.pool_by is not None:
        members = tuple(sorted(ctx.train_pool["id"].unique()))
        key = (ctx.pool_by, ctx.origin, members, tuple(sorted(overrides.items())))
        if key not in _pooled_models:
            _pooled_models[key] = _fit(x_train, y_train - level_train, inner_fit, params)
        model = _pooled_models[key]
    else:
        model = _fit(x_train, y_train - level_train, inner_fit, params)

    return np.clip(model.predict(x_pred) + level_pred, 0, None)
