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

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features import POOL_ID_COLS, supervised_feature_columns


@dataclass(frozen=True)
class Context:
    """
    Everything a forecaster is allowed to see when producing one forecast.

    Attributes
    ----------
    history
        The series' own observed sales, ending at (and including) `origin`.
    targets
        The days being forecast: dates and known-in-advance calendar columns
        **only**. Sales for these days are deliberately absent.
    origin
        The forecast origin T. Every prediction is for T+1 .. T+horizon.
    horizon
        Number of days being forecast.
    series_id
        Which series is being forecast. Needed because a pooled `train_pool`
        holds rows for every store, and only this one's are to be predicted.
    season
        Length of the seasonal cycle in days (7 for daily retail).
    train_pool
        Supervised rows a learned model may train on, already filtered so every
        row's target was observable at `origin`. `None` for benchmarks, which
        need no training set.
    pool_by
        `Config.pool_by`, passed through. None means `train_pool` holds this
        series alone; anything else means it spans several series and the
        model is given their identity columns as features.
    seed
        Fixed, so repeat runs on identical inputs give identical output.
    """

    history: pd.DataFrame
    targets: pd.DataFrame
    origin: pd.Timestamp
    horizon: int
    series_id: str

    season: int = 7
    train_pool: pd.DataFrame | None = None
    pool_by: str | None = None
    seed: int = 0

    @property
    def y(self) -> np.ndarray:
        """The observed sales history, as a float array."""
        return self.history["sales"].to_numpy(dtype=float)


Forecaster = Callable[[Context], np.ndarray]


def _flat(value: float, ctx: Context) -> np.ndarray:
    """A constant forecast repeated across the horizon."""
    return np.full(ctx.horizon, float(value))


# --------------------------------------------------------------------------- #
# Benchmarks - FPP Ch. 5
# --------------------------------------------------------------------------- #


def bench_mean(ctx: Context) -> np.ndarray:
    """
    Mean method: the average of all history.

    FPP §5.2. Unbeatable-looking on a stationary series and hopeless on a
    trending one, which is precisely why it belongs alongside the others.
    """
    return _flat(ctx.y.mean(), ctx)


def bench_naive(ctx: Context) -> np.ndarray:
    """
    Naive method: the last observed value, repeated.

    FPP §5.2, `ŷ_{T+h|T} = y_T`. Optimal for a random walk, and the benchmark
    any forecast of a genuinely unpredictable series should be measured against.
    """
    return _flat(ctx.y[-1], ctx)


def bench_seasonal_naive(ctx: Context) -> np.ndarray:
    """
    Seasonal naive: the value from the same weekday in the most recent week.

    FPP §5.2. Captures the weekly pattern for free and is what a person with no
    tools would do.

    Its weakness is the reason this project reports five benchmarks: it stakes
    everything on one specific past day, so its error carries that day's noise
    *plus* the target's. On a noisy series that is roughly sqrt(2) worse than
    predicting the mean, and a model can beat it by being sensibly dull.
    """
    y, s = ctx.y, ctx.season
    if len(y) < s:
        return _flat(y[-1], ctx)
    # h = 1..horizon maps back to the matching day of the last complete cycle.
    idx = [-s + ((h - 1) % s) for h in range(1, ctx.horizon + 1)]
    return y[idx]


def bench_drift(ctx: Context) -> np.ndarray:
    """
    Drift method: the last value, extrapolated along the average historical
    slope.

    FPP §5.2. Equivalent to drawing a straight line through the first and last
    observations and continuing it - a naive forecast that is allowed a trend.
    """
    y = ctx.y
    if len(y) < 2:
        return _flat(y[-1], ctx)
    slope = (y[-1] - y[0]) / (len(y) - 1)
    return y[-1] + slope * np.arange(1, ctx.horizon + 1)


def bench_moving_average(ctx: Context, window: int = 28) -> np.ndarray:
    """
    Flat forecast at the mean of the last `window` days.

    Not one of FPP's four, and included because it is the one that matters. It
    tracks the recent level rather than the whole history, and on this kind of
    data it is a genuinely hard benchmark to beat - a previous version of this
    project found a supposed 75% win rate collapse to 42% when measured against
    it instead of seasonal naive.
    """
    return _flat(ctx.y[-window:].mean(), ctx)


def bench_seasonal_naive_364(ctx: Context) -> np.ndarray:
    """
    Seasonal naive with a one-year period: the same weekday, 52 weeks ago.

    The weekly seasonal naive is what an orderer's default screen shows - this
    day last week. Before a holiday an experienced orderer switches to *this
    day last year*, and that is what this benchmark encodes. Lag 364 rather
    than 365 keeps the weekday aligned (FPP §13.1 on annual periods in daily
    data). Falls back to the weekly version when a year of history is not yet
    available, so the two benchmarks are identical on short series.

    Included so that the holiday-week comparison is against what a good orderer
    actually does, not only against the default screen.
    """
    y = ctx.y
    lag = 52 * ctx.season
    if len(y) < lag:
        return bench_seasonal_naive(ctx)
    idx = [-lag + (h - 1) for h in range(1, ctx.horizon + 1)]
    return y[idx]


BENCHMARKS: dict[str, Forecaster] = {
    "mean": bench_mean,
    "naive": bench_naive,
    "seasonal_naive": bench_seasonal_naive,
    "seasonal_naive_364": bench_seasonal_naive_364,
    "drift": bench_drift,
    "moving_average_28": bench_moving_average,
}


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

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


def fit_predict_ets(ctx: Context) -> np.ndarray:
    """
    Holt-Winters exponential smoothing with weekly seasonality (FPP Ch. 8).

    A real classical contender rather than a benchmark: a weighted average of
    the past where the weights decay exponentially, extended to carry a trend
    and a repeating weekly shape. On a smooth, strongly seasonal series it is
    often very hard to beat, which is exactly the case this study is about.

    Fitted on the most recent two years only. ETS weights recent observations
    most heavily anyway, and the full five years both slows the optimiser and
    drags the fitted seasonal shape toward a level the series left behind -
    step 3 showed a clear multi-year downward drift.

    If the optimiser fails, the forecast falls back to the 28-day mean and a
    warning names the series and origin. The fallback keeps a single bad fold
    from aborting a whole run; the warning keeps it from hiding - a silent
    fallback would let "ETS diverged every time" masquerade as "ETS is weak".
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    y = ctx.y[-730:]
    if len(y) < 2 * ctx.season:
        return _flat(ctx.y[-28:].mean(), ctx)

    try:
        fit = ExponentialSmoothing(
            y,
            trend="add",
            seasonal="add",
            seasonal_periods=ctx.season,
            initialization_method="estimated",
        ).fit()
        return np.clip(np.asarray(fit.forecast(ctx.horizon), dtype=float), 0, None)
    except Exception as err:
        warnings.warn(
            f"ETS failed for {ctx.series_id} at origin {ctx.origin.date()} "
            f"({type(err).__name__}: {err}); using the 28-day mean instead.",
            stacklevel=2,
        )
        return _flat(ctx.y[-28:].mean(), ctx)


# --- ARIMA ------------------------------------------------------------------

# Regressors handed to ARIMA. All are known in advance for any target date, so
# they are legitimate predictors in FPP's sense (§10.1). `pre_holiday` is the
# two-day run-up the event-effect table shows for this item.
ARIMA_EXOG = ("is_holiday", "pre_holiday", "snap")

# The differencing is fixed, not searched: no ordinary difference (the series
# is level-stationary over two years) and one seasonal difference at lag 7.
# FPP §9.7 is explicit that information criteria cannot compare models with
# different orders of differencing, so only the AR and MA orders are chosen.
ARIMA_D, ARIMA_SEASONAL_D = 0, 1
ARIMA_GRID = [
    (p, q, P, Q) for p in (0, 1, 2) for q in (0, 1, 2) for P in (0, 1) for Q in (0, 1)
]
ARIMA_FIT_DAYS = 730

# Chosen orders, one per series, filled on the first fold each series is
# forecast and reused after. The harness walks folds oldest first, so the
# selection is made on the earliest training window and never sees a scored
# day. Reset between unrelated runs with `arima_orders.clear()`.
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
    """AICc over ARIMA_GRID at fixed differencing. Returns (order, seasonal_order, aicc)."""
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
    Seasonal ARIMA with regressors (FPP Ch. 9, and Ch. 10 for the regressors).

    The classical model that *can* be told about the calendar. It is given the
    same known-in-advance information XGBoost gets - a holiday flag, a
    pre-holiday flag and the SNAP flag - so the comparison between the two is
    about the modelling, not about who was allowed to see the calendar.

    Fitted on the most recent two years, like ETS and for the same reason. The
    order is selected once per series by AICc on that series' first training
    window and then held fixed across folds: re-selecting on every fold would
    multiply the run time by the grid size and make fold-to-fold differences
    partly about which order happened to win.

    Fallback and warning follow ETS's pattern: a failed fit gives the 28-day
    mean, and says so.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    hist = ctx.history.iloc[-ARIMA_FIT_DAYS:]
    y = hist["sales"].to_numpy(dtype=float)
    if len(y) < 4 * ctx.season:
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
        return _flat(ctx.y[-28:].mean(), ctx)


MODELS: dict[str, Forecaster] = {
    "xgboost": fit_predict_xgboost,
    "ets": fit_predict_ets,
    "arima": fit_predict_arima,
}

# Everything that produces a forecast, benchmarks and models alike.
ALL_FORECASTERS: dict[str, Forecaster] = {**BENCHMARKS, **MODELS}
