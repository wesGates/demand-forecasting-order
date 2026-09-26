"""
Level-relative trees with recency weights on the training rows.

The fast item's level drifted year to year (109 a day in 2011 at the
busiest store, 82 in early 2016) and the slow item halved. Training on five
years with equal weight lets 2012 vote as loudly as last month. Here each
row is weighted by 0.5 ** (age in years), so a row from a year ago counts
half and one from three years ago an eighth. Everything else is the
level-relative model.
"""

from __future__ import annotations

import numpy as np

from src.models.base import Context
from src.models.xgboost_model import XGB_PARAMS, design
from src.models.xgboost_relative import LEVEL_FEATURE

HALF_LIFE_DAYS = 365

_pooled_models: dict[tuple, object] = {}


def reset() -> None:
    _pooled_models.clear()


def _fit_weighted(x_train, y_train, inner_fit, params, weight):
    from xgboost import XGBRegressor

    if inner_fit.sum() and (~inner_fit).sum():
        model = XGBRegressor(**params, enable_categorical=True, early_stopping_rounds=50, eval_metric="rmse")
        model.fit(
            x_train[inner_fit], y_train[inner_fit], sample_weight=weight[inner_fit],
            eval_set=[(x_train[~inner_fit], y_train[~inner_fit])], verbose=False,
        )
    else:
        model = XGBRegressor(**{**params, "n_estimators": 150}, enable_categorical=True)
        model.fit(x_train, y_train, sample_weight=weight)
    return model


def fit_predict_xgboost_rel_recent(ctx: Context, **overrides) -> np.ndarray:
    """Level-relative trees where older training rows count less."""
    parts = design(ctx)
    if parts is None:
        return np.full(ctx.horizon, np.nan)
    x_train, y_train, x_pred, inner_fit = parts

    # design() keeps the pool's row order, so the ages line up with x_train.
    train = ctx.train_pool[ctx.train_pool["target_date"] <= ctx.origin]
    age = (ctx.origin - train["target_date"]).dt.days.to_numpy(dtype=float)
    weight = 0.5 ** (age / HALF_LIFE_DAYS)

    level_train = x_train[LEVEL_FEATURE].to_numpy(dtype=float)
    level_pred = x_pred[LEVEL_FEATURE].to_numpy(dtype=float)
    params = {**XGB_PARAMS, **overrides, "random_state": ctx.seed}

    if ctx.pool_by is not None:
        members = tuple(sorted(ctx.train_pool["id"].unique()))
        key = (ctx.pool_by, ctx.origin, members, tuple(sorted(overrides.items())))
        if key not in _pooled_models:
            _pooled_models[key] = _fit_weighted(x_train, y_train - level_train, inner_fit, params, weight)
        model = _pooled_models[key]
    else:
        model = _fit_weighted(x_train, y_train - level_train, inner_fit, params, weight)

    return np.clip(model.predict(x_pred) + level_pred, 0, None)
