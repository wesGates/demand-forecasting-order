"""
The run registry: recording, back-filling, predecessor lookup and paired
comparison, on a synthetic panel in a temporary cache so no real run is
touched.
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from src import registry
from src.step1_problem import Config
from src.step5_evaluate import run_walk_forward
from src.suites import SUITES, suite_config, suite_of
from src.validate import _synthetic_panel


@pytest.fixture
def cfg(tmp_path):
    return Config(n_folds=6, min_train_days=200, cache_dir=tmp_path / "cache", data_dir=tmp_path)


@pytest.fixture
def panel():
    return _synthetic_panel(n_series=3, n_days=400)


def _record(cfg, panel, methods, note=None):
    """registry.record() without the M5 loader: run, then register."""
    con = registry.connect(cfg)
    ids = []
    for method in methods:
        predictions = run_walk_forward(panel, cfg, methods=[method], progress=False)
        row, scores = registry._run_row(
            cfg, method, predictions, source="run", recorded_at=f"2026-01-0{len(ids) + 1}T00:00:00+00:00",
            run_seconds=1.0, note=note, git=registry._git_state(), panel=panel,
        )
        registry._insert(con, row, scores)
        ids.append(row["run_id"])
    con.close()
    return ids


def test_record_writes_one_run_row_and_one_fold_score_per_store_origin(cfg, panel):
    (run_id,) = _record(cfg, panel, ["moving_average_28"], note="first")
    con = sqlite3.connect(registry.registry_path(cfg))
    run = pd.read_sql_query("SELECT * FROM run", con)
    folds = pd.read_sql_query("SELECT * FROM fold_score", con)
    con.close()
    assert list(run["run_id"]) == [run_id]
    assert run.loc[0, "method"] == "moving_average_28"
    assert run.loc[0, "kind"] == "benchmark"
    assert run.loc[0, "note"] == "first"
    assert run.loc[0, "code_current"] == 1
    assert len(folds) == 3 * cfg.n_folds  # three series, six origins
    assert folds["rmsse"].notna().all()
    # the headline row is the mean of the fold rows it summarises
    assert run.loc[0, "rmsse_mean"] == pytest.approx(folds["rmsse"].mean())


def test_model_runs_carry_paired_improvement_over_the_reference(cfg, panel):
    (run_id,) = _record(cfg, panel, ["ets"])
    row = registry.runs(cfg, "run_id = ?", (run_id,)).iloc[0]
    assert 0.0 <= row["win_vs_ref"] <= 1.0
    assert row["impr_q1"] <= row["impr_median"] <= row["impr_q3"]


def test_recording_twice_replaces_rather_than_duplicates(cfg, panel):
    _record(cfg, panel, ["moving_average_28"])
    _record(cfg, panel, ["moving_average_28"])
    con = sqlite3.connect(registry.registry_path(cfg))
    n_runs = con.execute("SELECT COUNT(*) FROM run").fetchone()[0]
    n_folds = con.execute("SELECT COUNT(*) FROM fold_score").fetchone()[0]
    con.close()
    assert n_runs == 1
    assert n_folds == 3 * cfg.n_folds


def test_backfill_indexes_cached_runs_and_is_idempotent(cfg, panel):
    for method in ("moving_average_28", "seasonal_naive"):
        run_walk_forward(panel, cfg, methods=[method], progress=False)  # writes the cache
    assert registry.backfill(cfg, verbose=False) == 2
    assert registry.backfill(cfg, verbose=False) == 0
    rows = registry.runs(cfg)
    assert set(rows["method"]) == {"moving_average_28", "seasonal_naive"}
    assert (rows["source"] == "backfill").all()


def test_predecessor_is_the_previous_run_of_the_same_method_and_layout(cfg, panel):
    first, = _record(cfg, panel, ["moving_average_28"])
    # a second registration under a different run id stands in for a code change
    con = registry.connect(cfg)
    row = registry.runs(cfg, "run_id = ?", (first,)).iloc[0].to_dict()
    row.update(run_id="moving_average_28_changed", recorded_at="2026-02-01T00:00:00+00:00")
    folds = pd.read_sql_query("SELECT * FROM fold_score WHERE run_id = ?", con, params=(first,))
    folds["rmsse"] = folds["rmsse"] * 0.9  # the "new" version is 10% better everywhere
    scores = folds.drop(columns=["run_id"])
    registry._insert(con, row, scores)
    con.close()

    assert registry.predecessor("moving_average_28_changed", cfg) == first
    assert registry.predecessor(first, cfg) is None
    c = registry.compare(first, "moving_average_28_changed", cfg)
    assert c["n_paired"] == 3 * cfg.n_folds
    assert c["win_rate_b"] == 1.0
    assert c["impr_median_pct"] == pytest.approx(10.0)


def test_suites_are_recognised_from_a_config():
    assert suite_of(suite_config("dev", "fast")) == "dev"
    assert suite_of(suite_config("weekly", "fast")) == "weekly"
    assert suite_of(suite_config("everyday", "slow", pool_by="item_id")) == "everyday"
    assert suite_of(Config(n_folds=104)) is None
    assert set(SUITES) == {"dev", "weekly", "everyday"}
