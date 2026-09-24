"""
Gradient-boosted trees on a level-relative target.

Same features and settings as the point model. The target changes to sales
minus the 28-day mean at the origin, and that mean is added back at
prediction time.

The "level" of a series is its recent average, how much it sells per day
right now. A tree model has no idea of level. It predicts from leaves grown
on the values it saw in training, so it cannot predict a value below the
lowest one it trained on. Item 4 showed that on the declining item
FOODS_1_021 the plain trees ran 60% high because the level kept falling out
from under them. Exponential smoothing carries a level state that updates
with every new day (FPP §8.1), wich is why ETS did not have the problem.

Here the 28-day mean carries the level. It is a feature the row already
has, computed at the row's own origin, so training and prediction use the
same anchor and nothing from after the origin gets in. The trees only have
to learn the weekly and holiday shape around it.

The per-store and pooled variants work exactly like the point model's,
including the one-fit-per-origin memo when pooled.
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
    """The point model's trees, fitted to sales minus the 28-day mean at the origin."""
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
