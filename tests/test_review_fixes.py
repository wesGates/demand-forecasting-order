"""
Regression tests for the 2026-09-23 review (item 8): closures imputed from
the past only, short series skipped, fallbacks recorded, unscored folds
counted, a complete daily grid required, cached runs checked against the
data, and the registry's guards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src import registry
from src.scoring import score_folds, summarise
from src.step1_problem import Config
from src.step2_data import assert_daily_grid, impute_closures
from src.step5_evaluate import run_walk_forward
from src.validate import _synthetic_panel


def _panel_with_closure():
    """One series, 100 days, a closure on day 70 with holiday windows around days 63 and 84."""
    dates = pd.date_range("2015-01-01", periods=100, freq="D")
    df = pd.DataFrame({"id": "S", "date": dates, "sales": np.arange(100, dtype=float)})
    df["closure"] = df["date"] == dates[70]
    df["days_to_holiday"] = 30
    df["days_since_holiday"] = 30
    df.loc[63, "days_to_holiday"] = 1  # inside a holiday window: must be skipped
    df.loc[84, "days_since_holiday"] = 0  # a week after: would only matter if forward weeks were used
    return df


def test_closures_are_imputed_from_preceding_weeks_only_skipping_holiday_windows():
    df = _panel_with_closure()
    out = impute_closures(df)
    # candidates: days 42, 49, 56, 63; day 63 is in a holiday window -> mean of 42, 49, 56
    assert out.loc[70, "sales"] == pytest.approx(np.mean([42, 49, 56]))
    # nothing after the closure was used: change every later day and the value must not move
    later = df.copy()
    later.loc[71:, "sales"] = 9999.0
    assert impute_closures(later).loc[70, "sales"] == out.loc[70, "sales"]


def test_a_series_too_short_for_one_origin_is_skipped_not_fatal(tmp_path):
    panel = _synthetic_panel(n_series=2, n_days=320)
    short = panel[panel["id"] == panel["id"].iloc[0]].tail(40)  # under MIN_HISTORY + horizon
    mixed = pd.concat([short, panel[panel["id"] != panel["id"].iloc[0]]], ignore_index=True)
    cfg = Config(n_folds=3, min_train_days=100, cache_dir=tmp_path / "c", data_dir=tmp_path)
    p = run_walk_forward(mixed, cfg, methods=["seasonal_naive", "xgboost"], progress=False, use_cache=False, n_jobs=1)
    assert set(p["id"]) == {panel["id"].iloc[-1]}  # only the long series is scored


def test_a_failed_fit_is_recorded_as_a_fallback(tmp_path, monkeypatch):
    panel = _synthetic_panel(n_series=1, n_days=320)
    cfg = Config(n_folds=3, min_train_days=200, cache_dir=tmp_path / "c", data_dir=tmp_path)

    def broken(*args, **kwargs):
        raise RuntimeError("no convergence")

    import statsmodels.tsa.holtwinters as holtwinters

    monkeypatch.setattr(holtwinters, "ExponentialSmoothing", broken)  # the model imports it at call time
    p = run_walk_forward(panel, cfg, methods=["ets", "seasonal_naive"], progress=False, use_cache=False, n_jobs=1)
    ets = p[p["method"] == "ets"]
    assert ets["fallback"].all()
    assert not p[p["method"] == "seasonal_naive"]["fallback"].any()
    table = summarise(score_folds(p)).set_index("method")
    assert table.loc["ets", "n_fallback"] == cfg.n_folds
    assert table.loc["seasonal_naive", "n_fallback"] == 0


def test_a_missing_forecast_is_unscored_not_a_loss():
    rows = []
    for method, fc in (("a", [np.nan] * 7), ("b", [1.0] * 7), ("seasonal_naive", [2.0] * 7)):
        for h in range(7):
            rows.append({"id": "S", "item_id": "I", "store_id": "S", "fold": 0, "origin": pd.Timestamp("2015-01-04"),
                         "week_kind": "normal", "method": method, "kind": "benchmark" if method == "seasonal_naive" else "model",
                         "horizon": h + 1, "target_date": pd.Timestamp("2015-01-05") + pd.Timedelta(days=h),
                         "actual": 1.0, "forecast": fc[h], "scale": 1.0, "closure": False, "holiday_window": False, "fallback": False})
    scores = score_folds(pd.DataFrame(rows)).set_index("method")
    assert np.isnan(scores.loc["a", "rmsse"]) and bool(scores.loc["a", "unscored"])
    assert not bool(scores.loc["b", "unscored"])


def test_a_gap_in_the_daily_grid_is_refused():
    panel = _synthetic_panel(n_series=1, n_days=100)
    assert_daily_grid(panel)  # complete: fine
    with pytest.raises(ValueError, match="gaps"):
        assert_daily_grid(panel.drop(index=[50]))


def test_a_cached_run_made_on_other_data_is_refused(tmp_path):
    panel = _synthetic_panel(n_series=1, n_days=320)
    cfg = Config(n_folds=3, min_train_days=200, cache_dir=tmp_path / "c", data_dir=tmp_path)
    run_walk_forward(panel, cfg, methods=["seasonal_naive"], progress=False)  # writes the cache
    changed = panel.copy()
    changed.loc[changed.index[-5], "sales"] += 100
    with pytest.raises(ValueError, match="does not match the data"):
        run_walk_forward(changed, cfg, methods=["seasonal_naive"], progress=False)


def test_registry_compare_skips_undefined_pairs_and_keeps_first_record(tmp_path):
    cfg = Config(n_folds=4, min_train_days=200, cache_dir=tmp_path / "c", data_dir=tmp_path)
    panel = _synthetic_panel(n_series=2, n_days=320)
    con = registry.connect(cfg)
    predictions = run_walk_forward(panel, cfg, methods=["moving_average_28"], progress=False, n_jobs=1)
    row, scores = registry._run_row(cfg, "moving_average_28", predictions, source="run",
                                    recorded_at="2026-01-01T00:00:00+00:00", run_seconds=1.0, note="first",
                                    git=registry._git_state(), panel=panel)
    assert registry._insert(con, row, scores)
    # a second recording of the same run: first row kept, note appended
    row2 = {**row, "recorded_at": "2026-02-01T00:00:00+00:00", "note": "again"}
    assert not registry._insert(con, row2, scores)
    got = registry.runs(cfg, "run_id = ?", (row["run_id"],)).iloc[0]
    assert got["recorded_at"] == "2026-01-01T00:00:00+00:00" and got["note"] == "first | again"
    # a run B with one zero-RMSSE fold in A: that pair is undefined, the rest compare
    b = {**row, "run_id": "other_run", "recorded_at": "2026-03-01T00:00:00+00:00", "note": None}
    zeroed = scores.copy()
    con.execute("UPDATE fold_score SET rmsse = 0 WHERE run_id = ? AND fold = 0", (row["run_id"],))
    con.commit()
    zeroed["rmsse"] = zeroed["rmsse"] * 0.5
    registry._insert(con, b, zeroed)
    con.close()
    c = registry.compare(row["run_id"], "other_run", cfg)
    assert c["n_undefined"] == 2  # two series, fold 0
    assert np.isfinite(c["impr_mean_pct"])
    assert c["impr_median_pct"] == pytest.approx(50.0)
