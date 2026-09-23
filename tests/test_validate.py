"""
The validator - the checks that need no M5 data.

README: `python -m src.validate` runs nine checks and the project refuses to
report a number until it passes. Checks 2, 4, 5 and 6 read the raw files,
which are not in the repository; they are not exercised here.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import validate
from src.step1_problem import Config


def test_run_all_lists_nine_checks():
    import inspect

    src = inspect.getsource(validate.run_all)
    names = [n for n in dir(validate) if n.startswith("check_")]
    assert len(names) == 9
    for name in names:
        assert name in src


def test_closed_form_benchmarks_pass():
    result = validate.check_benchmarks_closed_form()
    assert result["verdict"] == "ok", result["failures"]


def test_quantile_scoring_passes():
    result = validate.check_quantile_scoring()
    assert result["verdict"] == "ok", result["failures"]
    for tau, coverage in result["coverage"].items():
        assert abs(coverage - tau) <= 0.08


def test_synthetic_panel_has_the_documented_structure():
    p = validate._synthetic_panel(n_series=2, n_days=100, noise_sd=0.0)
    assert p["id"].nunique() == 2
    # Noise-free: the series is level + weekly shape + slow trend, so the same
    # weekday a week apart differs only by the trend increment.
    s = p[p["id"] == "SYNTH_00"].sort_values("date")["sales"].to_numpy()
    step = 3.0 / 99
    np.testing.assert_allclose(s[7:] - s[:-7], 7 * step, atol=1e-9)


@pytest.mark.slow
def test_noise_floor_and_shuffled_target_on_a_small_panel():
    """
    A trimmed version of checks 3 and 4: XGBoost on a synthetic panel must
    not beat the injected noise, and on a shuffled target must not beat the
    mean. Small so the whole suite stays quick.
    """
    cfg = Config(n_folds=3, min_train_days=200)
    panel = validate._synthetic_panel(n_series=2, n_days=420, noise_sd=2.0)
    observed, _, n = validate._walk_forward_rmse(panel, cfg)
    assert n == 6
    assert observed / 2.0 >= 0.90

    rng = np.random.default_rng(7)
    panel["sales"] = (
        panel.groupby("id")["sales"]
        .transform(lambda s: rng.permutation(s.to_numpy()))
        .to_numpy()
    )
    model, bench, _ = validate._walk_forward_rmse(panel, cfg)
    assert model / bench >= 0.95
