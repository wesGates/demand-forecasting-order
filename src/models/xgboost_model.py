"""Gradient-boosted trees on the supervised feature matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import POOL_ID_COLS, supervised_feature_columns
from src.models.base import Context

# Days of the training tail held out to decide when to stop adding trees.
INNER_VAL_DAYS = 90

XGB_PARAMS = dict(
    n_estimators=2000,  # an upper bound; early stopping picks the real number
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=-1,
    tree_method="hist",
)


def fit_predict_xgboost(ctx: Context, **overrides) -> np.ndarray:
    """
    Gradient-boosted trees on the supervised matrix.

    Identical code for both the per-series and the pooled variant - the only
    difference is how many stores' rows arrived in `ctx.train_pool`, and whether
    store identity is offered as a feature. Isolating the comparison to that one
    difference is the point.

    Predictions are clipped at zero: demand cannot be negative, and a tree
    ensemble extrapolating below the data would otherwise happily go there.
    """
    from xgboost import XGBRegressor

    pool = ctx.train_pool
    if pool is None or pool.empty:
        raise ValueError("xgboost needs a train_pool")

    cols = supervised_feature_columns(pool, pool_by=ctx.pool_by)

    # Trainable rows are those whose ANSWER was already observable at the
    # origin - i.e. target_date <= T. Filtering on origin_date instead would be
    # a leak: a row with origin T-1 has targets running to T+6, six days the
    # forecaster has not lived through yet.
    train = pool[pool["target_date"] <= ctx.origin]

    # Predict only for the series being forecast. A pooled train_pool holds
    # every store, so matching on origin alone would return one forecast per
    # store instead of one per horizon day.
    predict = pool[(pool["origin_date"] == ctx.origin) & (pool["id"] == ctx.series_id)]

    if train.empty or predict.empty:
        return np.full(ctx.horizon, np.nan)
    if (train["target_date"] > ctx.origin).any():
        raise ValueError("LEAK: training rows carry targets from after the origin.")

    x_train, x_pred = train[cols].copy(), predict[cols].copy()
    if ctx.pool_by is not None:
        # Series identity as native categoricals, so a pooled model can learn a
        # per-store or per-item level without one-hot columns. A column that
        # happens to be constant across this pool is harmless: the trees simply
        # never find a split on it.
        for col in POOL_ID_COLS:
            cats = pd.CategoricalDtype(sorted(pool[col].unique()))
            x_train[col] = x_train[col].astype(cats)
            x_pred[col] = x_pred[col].astype(cats)

    params = {**XGB_PARAMS, **overrides, "random_state": ctx.seed}

    # Early stopping on a chronological tail of the training data. Without it
    # the tree count is a guess: measured on this data, a fixed 400 trees fits
    # roughly three times more model than the data supports, and the extra is
    # memorisation (in-sample RMSE 4.1 against 12.8 held out).
    #
    # The split is by date, never at random - the inner validation set has to
    # sit *after* the rows used to fit, or stopping is decided by a model that
    # has already seen the period it is being judged on.
    cut = ctx.origin - pd.Timedelta(days=INNER_VAL_DAYS)
    inner_fit = train["target_date"] <= cut
    inner_val = ~inner_fit

    if inner_fit.sum() and inner_val.sum():
        model = XGBRegressor(
            **params,
            enable_categorical=True,
            early_stopping_rounds=50,
            eval_metric="rmse",
        )
        model.fit(
            x_train[inner_fit.to_numpy()],
            train.loc[inner_fit, "target"],
            eval_set=[(x_train[inner_val.to_numpy()], train.loc[inner_val, "target"])],
            verbose=False,
        )
    else:
        # Too little history to hold anything back - fall back to a fixed,
        # deliberately modest tree count rather than the 2000 upper bound.
        model = XGBRegressor(**{**params, "n_estimators": 150}, enable_categorical=True)
        model.fit(x_train, train["target"])

    return np.clip(model.predict(x_pred), 0, None)
