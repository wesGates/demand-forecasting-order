"""
The parallel harness must give the same forecasts as the in-process path,
row for row, for per-series and pooled configs, and for the method whose
state is chosen once per series (ARIMA). Synthetic panel, temporary cache.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.step1_problem import Config
from src.step5_evaluate import run_walk_forward
from src.validate import _synthetic_panel


@pytest.fixture
def cfg(tmp_path):
    return Config(n_folds=4, min_train_days=200, cache_dir=tmp_path / "cache", data_dir=tmp_path)


@pytest.fixture
def panel():
    return _synthetic_panel(n_series=3, n_days=320)


def _run(panel, cfg, methods, n_jobs):
    return run_walk_forward(panel, cfg, methods=methods, progress=False, use_cache=False, n_jobs=n_jobs)


def test_per_series_methods_match_between_one_and_three_workers(panel, cfg):
    methods = ["seasonal_naive", "moving_average_28", "ets", "xgboost", "xgboost_rel"]
    serial = _run(panel, cfg, methods, n_jobs=1)
    parallel = _run(panel, cfg, methods, n_jobs=3)
    assert_frame_equal(serial, parallel)
    assert set(serial["method"]) == set(methods)
    assert len(serial) == 3 * cfg.n_folds * len(methods) * cfg.horizon


def test_pooled_methods_match_between_one_and_three_workers(panel, cfg):
    pooled = Config(**{**cfg.__dict__, "pool_by": "item_id"})
    methods = ["seasonal_naive", "xgboost", "xgboost_rel"]
    serial = _run(panel, pooled, methods, n_jobs=1)
    parallel = _run(panel, pooled, methods, n_jobs=3)
    assert_frame_equal(serial, parallel)


def test_arima_order_is_chosen_once_per_series_and_matches(panel, cfg):
    two = panel[panel["id"].isin(sorted(panel["id"].unique())[:2])]
    serial = _run(two, cfg, ["arima"], n_jobs=1)
    parallel = _run(two, cfg, ["arima"], n_jobs=2)
    assert_frame_equal(serial, parallel)


def test_parallelism_is_not_part_of_the_cache_key(panel, cfg):
    from src.step5_evaluate import _method_cache_path

    assert _method_cache_path(cfg, "ets") == _method_cache_path(cfg, "ets")
    a = run_walk_forward(panel, cfg, methods=["ets"], progress=False, n_jobs=1)
    b = run_walk_forward(panel, cfg, methods=["ets"], progress=False, n_jobs=3)  # served from cache
    assert_frame_equal(a, b)
