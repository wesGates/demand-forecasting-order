"""Gradient-boosted trees (XGBoost) on the supervised feature matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import POOL_ID_COLS, supervised_feature_columns
from src.models.base import Context

# The last 90 days of each training window are held out to decide when to
# stop adding trees.
INNER_VAL_DAYS = 90

XGB_PARAMS = dict(
    n_estimators=2000,  # an upper bound, early stopping picks the real count
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=1,  # one thread per fit. The harness runs many fits at once instead
    tree_method="hist",
)


def design(ctx: Context):
    """
    The training and prediction matrices for one fold, plus the early-stopping
    split. The point, level-relative and quantile models all use this, so they
    differ only in what they are asked to predict.

    Returns (x_train, y_train, x_pred, inner_fit), or None when there is
    nothing to train or predict. `inner_fit` marks the rows used to fit;
    the rest are the held-out tail.
    """
    pool = ctx.train_pool
    if pool is None or pool.empty:
        raise ValueError("xgboost needs a train_pool")

    cols = supervised_feature_columns(pool, pool_by=ctx.pool_by)

    # A row is trainable when its target was already known at the origin, so
    # target_date <= T. Filtering on origin_date would leak: a row with origin
    # T-1 has targets out to T+6, and the forecaster has not seen those days.
    train = pool[pool["target_date"] <= ctx.origin]

    # Predict for this series only. A pooled train_pool holds every store, and
    # matching on the origin alone would give one forecast per store.
    predict = pool[(pool["origin_date"] == ctx.origin) & (pool["id"] == ctx.series_id)]

    if train.empty or predict.empty:
        return None
    if (train["target_date"] > ctx.origin).any():
        raise ValueError("LEAK: training rows carry targets from after the origin.")

    x_train, x_pred = train[cols].copy(), predict[cols].copy()
    if ctx.pool_by is not None:
        # Store and item ids as categoricals, so a pooled model can learn a
        # level per store without one-hot columns. A column that is constant
        # across the pool is harmless, the trees never split on it.
        for col in POOL_ID_COLS:
            cats = pd.CategoricalDtype(sorted(pool[col].unique()))
            x_train[col] = x_train[col].astype(cats)
            x_pred[col] = x_pred[col].astype(cats)

    # Early stopping on the last 90 days of the training window. Without it
    # the tree count is a guess. A fixed 400 trees fit about three times more
    # model than the data supports and the extra was memorisation (in-sample
    # RMSE 4.1 against 12.8 held out).
    #
    # The split is by date. The validation rows have to come after the rows
    # used to fit, or the stopping decision is made by a model that already
    # saw the period it is judged on.
    cut = ctx.origin - pd.Timedelta(days=INNER_VAL_DAYS)
    inner_fit = (train["target_date"] <= cut).to_numpy()
    return x_train, train["target"].to_numpy(dtype=float), x_pred, inner_fit


# Fitted pooled models for the current task, keyed by (pool scope, origin,
# members). A pooled model trains on the whole pool, so it is the same model
# whichever store asks for it. Known mistake: the first pooled run had no
# memo and fitted the same model ten times per origin, once per store. lol.
# Cleared by `reset_run_state()` and by the harness at the start of each task.
_pooled_models: dict[tuple, object] = {}


def reset() -> None:
    _pooled_models.clear()


def _fit(x_train, y_train, inner_fit, params):
    from xgboost import XGBRegressor

    if inner_fit.sum() and (~inner_fit).sum():
        model = XGBRegressor(
            **params,
            enable_categorical=True,
            early_stopping_rounds=50,
            eval_metric="rmse",
        )
        model.fit(
            x_train[inner_fit],
            y_train[inner_fit],
            eval_set=[(x_train[~inner_fit], y_train[~inner_fit])],
            verbose=False,
        )
    else:
        # Too little history to hold a tail back. Use a fixed, modest tree
        # count instead of the 2000 upper bound.
        model = XGBRegressor(**{**params, "n_estimators": 150}, enable_categorical=True)
        model.fit(x_train, y_train)
    return model


def fit_predict_xgboost(ctx: Context, **overrides) -> np.ndarray:
    """
    Gradient-boosted trees on the supervised matrix.

    The same code serves the per-store and the pooled variant. The only
    difference is how many stores' rows are in `ctx.train_pool` and whether
    store identity is a feature, which keeps the pooling comparison down to
    that one change. A pooled model is fitted once per origin and shared.

    Predictions are clipped at zero, because demand cannot be negative and
    the trees would otherwise go there.
    """
    parts = design(ctx)
    if parts is None:
        return np.full(ctx.horizon, np.nan)
    x_train, y_train, x_pred, inner_fit = parts
    params = {**XGB_PARAMS, **overrides, "random_state": ctx.seed}

    if ctx.pool_by is not None:
        members = tuple(sorted(ctx.train_pool["id"].unique()))
        key = (ctx.pool_by, ctx.origin, members, tuple(sorted(overrides.items())))
        if key not in _pooled_models:
            _pooled_models[key] = _fit(x_train, y_train, inner_fit, params)
        return np.clip(_pooled_models[key].predict(x_pred), 0, None)

    model = _fit(x_train, y_train, inner_fit, params)
    return np.clip(model.predict(x_pred), 0, None)
