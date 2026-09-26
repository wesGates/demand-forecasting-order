"""
Step 4 - Choosing and fitting models (FPP §1.6, step 4).

Everything that produces a forecast lives here, behind one interface:

    forecaster(ctx: Context) -> np.ndarray of length ctx.horizon

`Context` carries everything a forecaster may legitimately see. Crucially it
carries **no future information at all** - the target values for the days being
forecast are not in it, so a benchmark cannot accidentally read the answer. That
is a deliberate change from the previous build, where a baseline received the
test frame and one of them quietly indexed into it.

Two registries, both flat dicts: adding a method is a function plus an entry.

**Benchmarks** (FPP Ch. 5) are what every model must beat to be interesting.
All five are here, not one. A single benchmark has already misled this project
once: seasonal naive predicts one specific past day, so on a noisy series its
error runs about sqrt(2) worse than simply predicting the mean, and a model that
predicts near the mean "wins" without any skill at all. The mean-based methods
are the guard against that.

**Models** are the candidates under test:

  xgboost   gradient-boosted trees on the supervised feature matrix
  ets       Holt-Winters exponential smoothing (FPP Ch. 8)
  arima     seasonal ARIMA with holiday and SNAP regressors (FPP Ch. 9-10)

ETS and ARIMA are the two classical families FPP puts forward, and they are
not interchangeable here: the ETS implementation takes no regressors, so it
cannot be told a holiday is coming, while ARIMA can. Keeping both shows what
the calendar information is worth to a classical model. A plain ARMA is not in
the set on purpose - it is ARIMA without differencing or a seasonal term, and on
daily data with a weekly cycle it would need absurd orders to imitate the
seasonality that one seasonal term captures.

**Pooling is not a model - it is a training scope**, and it lives on
`Config.pool_by`. "XGBoost per store" and "XGBoost pooled across stores" are the
same function with the same features; the only difference is which rows step 5
puts into `Context.train_pool`, and whether the model is handed series identity
to tell them apart. Keeping that on one switch is what isolates the
cross-learning question to a single variable.
"""

from __future__ import annotations

from src.models import (
    arima_model,
    arma_model,
    benchmarks,
    ets_model,
    xgboost_model,
    xgboost_quantile,
    xgboost_poisson,
    xgboost_rel_recent,
    xgboost_relative,
)
from src.models.arima_model import (
    _arima_exog,
    _select_arima_order,
    arima_orders,
    fit_predict_arima,
)
from src.models.base import Context, Forecaster, _flat
from src.models.benchmarks import BENCHMARKS
from src.models.ets_model import fit_predict_ets
from src.models.xgboost_model import INNER_VAL_DAYS, XGB_PARAMS, fit_predict_xgboost

MODELS: dict[str, Forecaster] = {
    "xgboost": fit_predict_xgboost,
    "ets": fit_predict_ets,
    "arima": fit_predict_arima,
    "xgboost_rel": xgboost_relative.fit_predict_xgboost_relative,
    "xgboost_poisson": xgboost_poisson.fit_predict_xgboost_poisson,
    "xgboost_rel_recent": xgboost_rel_recent.fit_predict_xgboost_rel_recent,
    "arima_plain": arma_model.fit_predict_arima_plain,
    "arma": arma_model.fit_predict_arma,
    **xgboost_quantile.QUANTILE_MODELS,
}

# Everything that produces a forecast, benchmarks and models alike.
ALL_FORECASTERS: dict[str, Forecaster] = {**BENCHMARKS, **MODELS}

# Which module each method's code lives in. The predictions cache is keyed
# per method on the code of *that* module (plus the shared base and harness),
# so editing XGBoost does not throw away a cached ARIMA run.
MODULE_OF = {
    **{name: benchmarks for name in BENCHMARKS},
    "xgboost": xgboost_model,
    "ets": ets_model,
    "arima": arima_model,
    "xgboost_rel": xgboost_relative,
    "xgboost_poisson": xgboost_poisson,
    "xgboost_rel_recent": xgboost_rel_recent,
    "arima_plain": arma_model,
    "arma": arma_model,
    **{name: xgboost_quantile for name in xgboost_quantile.QUANTILE_MODELS},
}


def reset_run_state() -> None:
    """
    Forget everything memoised within a run: ARIMA's chosen orders and the
    quantile model's per-fold fits. The harness calls this at the start of
    every run so a second layout in the same process starts clean.
    """
    arima_orders.clear()
    xgboost_quantile.reset()
    xgboost_model.reset()
    xgboost_relative.reset()
    xgboost_rel_recent.reset()
    arma_model.reset()


__all__ = [
    "ALL_FORECASTERS",
    "BENCHMARKS",
    "Context",
    "Forecaster",
    "INNER_VAL_DAYS",
    "MODELS",
    "MODULE_OF",
    "XGB_PARAMS",
    "_arima_exog",
    "_flat",
    "_select_arima_order",
    "arima_orders",
    "reset_run_state",
    "fit_predict_arima",
    "fit_predict_ets",
    "fit_predict_xgboost",
]
