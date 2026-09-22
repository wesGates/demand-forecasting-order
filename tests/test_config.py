"""
Step 1 - the fold layout `Config` promises.

README: 52 walk-forward folds of 7 days, one full year, 25 May 2015 to
22 May 2016; folds tile the held-out period with no gaps and no overlap, and
`holdout_start` is the single source of truth step 3 and step 5 share.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.step1_problem import Config

LAST_M5_DATE = pd.Timestamp("2016-05-22")


def test_default_layout_scores_the_year_the_readme_quotes():
    cfg = Config()
    assert (cfg.n_folds, cfg.horizon, cfg.test_window) == (52, 7, 7)
    assert cfg.holdout_start(LAST_M5_DATE) == pd.Timestamp("2015-05-25")
    assert cfg.test_days_total == 364


def test_folds_tile_the_holdout_exactly():
    cfg = Config(n_folds=5, horizon=7)
    last = pd.Timestamp("2016-05-22")
    origins = cfg.fold_origins(last)
    start = cfg.holdout_start(last)

    assert len(origins) == 5
    # The first origin is the day before the first scored day.
    assert origins[0] == start - pd.Timedelta(days=1)
    # Consecutive origins are exactly one window apart: no gaps, no overlap.
    assert all(b - a == pd.Timedelta(days=7) for a, b in zip(origins, origins[1:]))
    # The last fold's window ends on the final day of data.
    assert origins[-1] + pd.Timedelta(days=cfg.test_window) == last


def test_every_origin_is_the_same_weekday():
    # OPEN_QUESTIONS #2: origins step by 7, so they all land on one weekday.
    cfg = Config(n_folds=10)
    origins = cfg.fold_origins(LAST_M5_DATE)
    assert len({o.dayofweek for o in origins}) == 1
    assert origins[0].day_name() == "Sunday"


def test_rmsse_lag_defaults_to_season_and_window_to_horizon():
    cfg = Config(season=7)
    assert cfg.rmsse_scale_lag == 7
    assert cfg.test_window == 7
    assert Config(season=7, rmsse_scale_lag=1).rmsse_scale_lag == 1


@pytest.mark.parametrize(
    "bad",
    [
        dict(test_window=28),  # the non-negotiable safeguard
        dict(horizon=0),
        dict(n_folds=0),
        dict(rmsse_scale_lag=0),
        dict(pool_by="item"),
        dict(rmsse_scale_window="all"),
    ],
)
def test_bad_configs_are_rejected(bad):
    with pytest.raises(ValueError):
        Config(**bad)


def test_config_is_frozen():
    cfg = Config()
    with pytest.raises(Exception):
        cfg.horizon = 14  # type: ignore[misc]


def test_relative_paths_are_anchored_to_the_repo_root(tmp_path):
    cfg = Config()
    assert cfg.data_dir.is_absolute()
    assert cfg.data_dir.name == "data"
    absolute = Config(data_dir=tmp_path)
    assert absolute.data_dir == tmp_path
