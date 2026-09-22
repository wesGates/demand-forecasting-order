"""
The quantile-objective XGBoost and the daily quantile scoring.

- one fit per (series, origin) serves every τ, and the run reset forgets it;
- quantiles are ordered per day and clipped at zero;
- the calibrated daily quantiles add the per-horizon error quantile from the
  calibration folds only, and the fitted ones pass through unchanged.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.models import xgboost_quantile as xq
from src.order import (
    daily_calibrated_quantiles,
    daily_native_quantiles,
    score_daily_quantiles,
    summarise_daily,
)
from src.step1_problem import Config
from src.step4_models import ALL_FORECASTERS, MODULE_OF, reset_run_state
from src.step5_evaluate import run_walk_forward
from tests.conftest import make_panel


def test_quantile_methods_are_registered_with_their_module():
    for tau in xq.TAUS:
        name = xq.method_name(tau)
        assert name in ALL_FORECASTERS
        assert MODULE_OF[name] is xq


def test_one_fit_serves_every_tau_and_reset_forgets_it(tmp_path):
    panel = make_panel(stores={"S1": ("CA", 50.0)})
    cfg = Config(n_folds=3, min_train_days=200, cache_dir=tmp_path)
    names = [xq.method_name(t) for t in xq.TAUS]
    p = run_walk_forward(panel, cfg, methods=names, progress=False, use_cache=False)
    assert len(xq._predicted) == 3  # one entry per fold, not one per tau
    assert set(p["method"]) == set(names)
    reset_run_state()
    assert xq._predicted == {}


def test_quantiles_are_ordered_and_non_negative(tmp_path):
    panel = make_panel(stores={"S1": ("CA", 50.0)})
    cfg = Config(n_folds=3, min_train_days=200, cache_dir=tmp_path)
    names = [xq.method_name(t) for t in xq.TAUS]
    p = run_walk_forward(panel, cfg, methods=names, progress=False, use_cache=False)
    wide = p.pivot_table(index=["origin", "horizon"], columns="method", values="forecast")
    assert (wide[names[0]] <= wide[names[1]]).all()
    assert (wide[names[1]] <= wide[names[2]]).all()
    assert (wide[names[2]] <= wide[names[3]]).all()
    assert (wide >= 0).all().all()


def _daily_frame():
    """Two folds calibration, one scored; errors +1 at h=1 and +5 at h=2 in calibration."""
    rows = []
    origins = pd.to_datetime(["2015-01-04", "2015-01-11", "2015-01-18"])
    for k, o in enumerate(origins):
        for h in (1, 2):
            err = {1: 1.0, 2: 5.0}[h] if k < 2 else 0.0
            rows.append(
                {
                    "id": "S",
                    "item_id": "I",
                    "store_id": "S",
                    "fold": k,
                    "origin": o,
                    "week_kind": "normal",
                    "horizon": h,
                    "target_date": o + pd.Timedelta(days=h),
                    "method": "ets",
                    "kind": "model",
                    "actual": 10.0 + err,
                    "forecast": 10.0,
                    "scale": 1.0,
                    "closure": False,
                    "holiday_window": False,
                }
            )
    return pd.DataFrame(rows)


def test_daily_calibration_is_per_horizon_from_calibration_folds_only():
    f = _daily_frame()
    scored_from = pd.Timestamp("2015-01-18")
    q = daily_calibrated_quantiles(f, scored_from, methods=("ets",), taus=(0.5,))
    assert set(q["fold"]) == {2}  # scored fold only
    by_h = q.set_index("horizon")["q"]
    assert by_h[1] == pytest.approx(11.0)  # forecast 10 + median error 1 at h=1
    assert by_h[2] == pytest.approx(15.0)  # forecast 10 + median error 5 at h=2
    assert (q["source"] == "calibrated").all()


def test_fitted_quantiles_pass_through_unchanged():
    f = _daily_frame().assign(method="xgboost_q50", forecast=12.5)
    q = daily_native_quantiles(f, pd.Timestamp("2015-01-18"))
    assert (q["q"] == 12.5).all() and (q["method"] == "xgboost_q").all()
    assert (q["tau"] == 0.5).all() and (q["source"] == "fitted").all()


def test_daily_scoring_and_summary_shapes():
    f = _daily_frame()
    q = daily_calibrated_quantiles(
        f, pd.Timestamp("2015-01-18"), methods=("ets",), taus=(0.5, 0.9)
    )
    s = score_daily_quantiles(q)
    assert {"pinball", "pinball_rel", "covered", "shortfall", "surplus"} <= set(s.columns)
    summ = summarise_daily(s)
    assert set(summ["tau"]) == {0.5, 0.9}
    assert (summ["coverage"] <= 1).all()
