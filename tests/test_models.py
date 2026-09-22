"""
Step 4 - the learned models' leakage guards, caches and fallbacks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.step4_models as m4
from src.features import build_supervised
from src.step1_problem import Config
from src.step4_models import Context, fit_predict_xgboost
from src.step5_evaluate import _supervised_path, run_walk_forward, supervised_matrices
from tests.conftest import make_panel


@pytest.fixture
def series():
    p = make_panel(n_days=400, stores={"S1": ("CA", 50.0)}, noise_sd=5.0, seed=2)
    return p.sort_values("date").reset_index(drop=True)


def _ctx(series, origin, pool, **kw):
    history = series[series["date"] <= origin]
    targets = series[
        (series["date"] > origin) & (series["date"] <= origin + pd.Timedelta(days=7))
    ]
    return Context(
        history=history,
        targets=targets.drop(columns=["sales"]),
        origin=origin,
        horizon=7,
        series_id=series["id"].iloc[0],
        train_pool=pool,
        **kw,
    )


def test_xgboost_trains_only_on_rows_whose_target_was_observable(series, monkeypatch):
    pool = build_supervised(series, 7)
    origin = series["date"].iloc[-8]
    seen = {}

    class Spy:
        def __init__(self, **kw):
            pass

        def fit(self, x, y, **kw):
            seen["n"] = len(x)
            eval_set = kw.get("eval_set")
            if eval_set:
                seen["n"] += len(eval_set[0][0])
            return self

        def predict(self, x):
            return np.zeros(len(x))

    import xgboost

    monkeypatch.setattr(xgboost, "XGBRegressor", Spy)
    out = fit_predict_xgboost(_ctx(series, origin, pool))
    assert out.shape == (7,)
    assert seen["n"] == int((pool["target_date"] <= origin).sum())


def test_xgboost_predicts_only_the_requested_series_from_a_pooled_matrix(
    series, monkeypatch
):
    other = series.copy()
    other["id"] = "ITEM_S2_evaluation"
    other["store_id"] = "S2"
    pool = pd.concat(
        [build_supervised(series, 7), build_supervised(other, 7)], ignore_index=True
    )
    origin = series["date"].iloc[-8]
    seen = {}

    class Spy:
        def __init__(self, **kw):
            pass

        def fit(self, x, y, **kw):
            seen["train_ids"] = set(x["store_id"].astype(str))
            return self

        def predict(self, x):
            seen["n_pred"] = len(x)
            return np.zeros(len(x))

    import xgboost

    monkeypatch.setattr(xgboost, "XGBRegressor", Spy)
    out = fit_predict_xgboost(_ctx(series, origin, pool, pool_by="item_id"))
    assert out.shape == (7,)
    assert seen["n_pred"] == 7  # one row per horizon day, not one per store
    assert seen["train_ids"] == {"S1", "S2"}  # but trained on the whole pool


def test_xgboost_returns_nan_when_the_origin_is_not_in_the_pool(series):
    pool = build_supervised(series, 7)
    origin = series["date"].iloc[-8] + pd.Timedelta(days=1)  # not an origin row
    out = fit_predict_xgboost(_ctx(series, origin, pool))
    assert np.isnan(out).all()


def test_xgboost_is_deterministic(series):
    pool = build_supervised(series, 7)
    origin = series["date"].iloc[-8]
    ctx = _ctx(series, origin, pool, seed=0)
    a, b = fit_predict_xgboost(ctx), fit_predict_xgboost(ctx)
    np.testing.assert_array_equal(a, b)
    assert (a >= 0).all()


def test_xgboost_without_a_pool_raises(series):
    with pytest.raises(ValueError, match="train_pool"):
        fit_predict_xgboost(_ctx(series, series["date"].iloc[-8], None))


def test_supervised_cache_key_tracks_feature_version(cfg, monkeypatch):
    import src.step5_evaluate as s5

    before = _supervised_path(cfg, "S")
    monkeypatch.setattr(s5, "FEATURE_VERSION", s5.FEATURE_VERSION + 1)
    assert _supervised_path(cfg, "S") != before


def test_supervised_cache_is_not_served_across_a_panel_version_bump(cfg, monkeypatch):
    """
    BUG: the predictions cache key includes `step2_data.CACHE_VERSION`, so a
    loader change forces a rerun. The supervised-matrix cache key does not:
    after bumping CACHE_VERSION (as v2 did, when closure imputation changed
    the sales the lags and targets are built from) the rerun trains XGBoost
    on matrices built from the *old* panel.
    """
    import src.step2_data as s2

    old = make_panel(n_days=300, stores={"S1": ("CA", 50.0)})
    new = old.copy()
    new["sales"] = new["sales"] * 2  # what a loader change looks like

    first = supervised_matrices(old, cfg)["ITEM_S1_evaluation"]
    monkeypatch.setattr(s2, "CACHE_VERSION", s2.CACHE_VERSION + 1)
    second = supervised_matrices(new, cfg)["ITEM_S1_evaluation"]

    assert not np.allclose(first["target"], second["target"]), (
        "supervised matrix served from cache after the panel version changed"
    )


def test_ets_falls_back_on_very_short_history(series):
    short = series.head(20)
    origin = short["date"].iloc[-8]
    out = m4.fit_predict_ets(_ctx(short, origin, None))
    np.testing.assert_allclose(
        out, np.full(7, short[short["date"] <= origin]["sales"].tail(28).mean())
    )


def test_arima_exog_marks_the_two_day_run_up():
    frame = pd.DataFrame(
        {
            "is_holiday": [0, 0, 0, 1, 0],
            "days_to_holiday": [3, 2, 1, 0, 30],
            "snap": [1, 0, 0, 1, 0],
        }
    )
    x = m4._arima_exog(frame)
    np.testing.assert_array_equal(x[:, 0], [0, 0, 0, 1, 0])  # is_holiday
    np.testing.assert_array_equal(x[:, 1], [0, 1, 1, 0, 0])  # pre_holiday
    np.testing.assert_array_equal(x[:, 2], [1, 0, 0, 1, 0])  # snap


class _FakeFit:
    aicc = 1.0

    def forecast(self, steps, exog=None):
        return np.full(steps, 1.0)


class _FakeSarimax:
    def __init__(self, y, exog=None, order=None, seasonal_order=None):
        pass

    def fit(self, **kw):
        return _FakeFit()


def test_arima_order_is_selected_on_this_runs_first_training_window(
    panel, tmp_path, monkeypatch
):
    """
    BUG: `fit_predict_arima` documents that the order is chosen "once per
    series by AICc on that series' first training window" and "never sees a
    scored day". The choice is memoised in a module-level dict that
    `run_walk_forward` never clears, so a second run in the same process
    (notebook 03's 104-fold run after notebook 02's 52-fold run, say) reuses
    an order chosen on a *different* layout's first window - for the longer
    layout, a window that ends inside its own scored period.
    """
    import statsmodels.tsa.statespace.sarimax as sm

    monkeypatch.setattr(sm, "SARIMAX", _FakeSarimax)
    windows = []

    def fake_select(y, exog, season):
        windows.append(len(y))
        return ((0, 0, 0), (0, 1, 0, season), 1.0)

    import src.models.arima_model as arima_model

    monkeypatch.setattr(arima_model, "_select_arima_order", fake_select)
    m4.arima_orders.clear()
    one_store = panel[panel["store_id"] == "S_BIG"]

    short = Config(n_folds=2, min_train_days=200, cache_dir=tmp_path)
    run_walk_forward(one_store, short, methods=["arima"], progress=False, use_cache=False)
    assert len(windows) == 1
    first_window_short = windows[0]

    longer = Config(n_folds=6, min_train_days=200, cache_dir=tmp_path)
    run_walk_forward(
        one_store, longer, methods=["arima"], progress=False, use_cache=False
    )
    m4.arima_orders.clear()

    assert len(windows) == 2, "second run reused the order chosen for another fold layout"
    assert windows[1] == first_window_short - 4 * 7
