"""
The level-relative XGBoost: same trees, target expressed as sales minus the
origin's 28-day mean. On a series whose level falls through the scored
period, its bias must be well below the plain point model's; on a flat
series the two must agree closely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.step1_problem import Config
from src.step4_models import MODULE_OF
from src.step5_evaluate import run_walk_forward, score_folds
from tests.conftest import make_panel


def _declining(panel: pd.DataFrame, floor: float = 0.3) -> pd.DataFrame:
    """Scale every store's sales down linearly to `floor` of its start over the panel."""
    p = panel.copy()
    t = (p["date"] - p["date"].min()).dt.days / (p["date"].max() - p["date"].min()).days
    p["sales"] = (p["sales"] * (1 - (1 - floor) * t)).round().clip(lower=0)
    return p


def test_relative_target_is_registered():
    from src.models import xgboost_relative

    assert MODULE_OF["xgboost_rel"] is xgboost_relative


def test_relative_target_tracks_a_falling_level(tmp_path):
    panel = _declining(
        make_panel(stores={"S1": ("CA", 60.0), "S2": ("TX", 40.0)}, noise_sd=2.0)
    )
    cfg = Config(n_folds=6, min_train_days=200, cache_dir=tmp_path)
    p = run_walk_forward(
        panel, cfg, methods=["xgboost", "xgboost_rel"], progress=False, use_cache=False
    )
    bias = score_folds(p).groupby("method")["bias"].mean()
    assert abs(bias["xgboost_rel"]) < abs(bias["xgboost"])
    assert bias["xgboost"] > 0  # the plain trees over-forecast a falling level


def test_relative_and_plain_agree_on_a_flat_series(tmp_path):
    panel = make_panel(stores={"S1": ("CA", 50.0)}, noise_sd=2.0)
    cfg = Config(n_folds=4, min_train_days=200, cache_dir=tmp_path)
    p = run_walk_forward(
        panel, cfg, methods=["xgboost", "xgboost_rel"], progress=False, use_cache=False
    )
    rm = score_folds(p).groupby("method")["rmse"].mean()
    assert np.isclose(rm["xgboost"], rm["xgboost_rel"], rtol=0.25)
