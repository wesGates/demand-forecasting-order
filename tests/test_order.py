"""
From forecast to order quantity (FPP §5.5, §5.9).

Documented behaviour under test (order.py docstrings, README):
  - weekly totals drop closure days; error = actual - forecast, positive is a
    shortfall;
  - pinball with the factor of two: tau = 0.5 gives |error|; shortfall costs
    2*tau per unit, surplus 2*(1-tau);
  - calibration uses only folds whose origin is before `scored_from`, scoring
    only folds at or after it; q = forecast + tau-quantile of the errors;
  - the expanding window uses every fold before the one being scored;
  - coverage = share of weeks with actual <= q; pinball_rel divides by the
    store's mean weekly sales over the scored period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.order import (
    calibrate,
    calibrate_expanding,
    pinball,
    quantile_forecasts,
    ranking_by_tau,
    score_quantiles,
    summarise_quantiles,
    weekly_totals,
)


def _predictions(
    errors_by_method: dict[str, list[float]], closure_fold: int | None = None
):
    """One series; each method's weekly error list becomes seven daily rows per fold."""
    rows = []
    for method, errors in errors_by_method.items():
        for fold, err in enumerate(errors):
            origin = pd.Timestamp("2014-01-05") + pd.Timedelta(days=7 * fold)
            for h in range(1, 8):
                closure = closure_fold == fold and h == 3
                rows.append(
                    {
                        "id": "S",
                        "item_id": "I",
                        "store_id": "S",
                        "fold": fold,
                        "origin": origin,
                        "week_kind": "holiday" if fold % 4 == 0 else "normal",
                        "method": method,
                        "kind": "model",
                        "horizon": h,
                        "target_date": origin + pd.Timedelta(days=h),
                        "actual": 100.0 + err / 7 if not closure else 0.0,
                        "forecast": 100.0,
                        "scale": 1.0,
                        "closure": closure,
                        "holiday_window": False,
                    }
                )
    return pd.DataFrame(rows)


def test_weekly_totals_sum_the_week_and_sign_the_error_as_shortfall():
    w = weekly_totals(_predictions({"m": [14.0, -7.0]}))
    assert len(w) == 2
    assert list(w["n_days"]) == [7, 7]
    np.testing.assert_allclose(w["actual"], [714.0, 693.0])
    np.testing.assert_allclose(w["forecast"], [700.0, 700.0])
    np.testing.assert_allclose(w["error"], [14.0, -7.0])  # actual - forecast


def test_weekly_totals_drop_closure_days_from_both_sides():
    w = weekly_totals(_predictions({"m": [0.0]}, closure_fold=0))
    assert w["n_days"].iloc[0] == 6
    assert w["forecast"].iloc[0] == pytest.approx(600.0)
    assert w["actual"].iloc[0] == pytest.approx(600.0)


def test_pinball_closed_form():
    y = np.array([10.0, 0.0, 5.0])
    q = np.array([0.0, 10.0, 5.0])
    np.testing.assert_allclose(pinball(y, q, 0.5), np.abs(y - q))
    np.testing.assert_allclose(pinball(y, q, 0.9), [18.0, 2.0, 0.0])
    np.testing.assert_allclose(pinball(y, q, 0.3), [6.0, 14.0, 0.0])


def test_calibrate_uses_only_folds_before_scored_from():
    errors = list(range(-10, 10))  # 20 folds, errors -10..9
    w = weekly_totals(_predictions({"m": [float(e) for e in errors]}))
    scored_from = w.sort_values("origin")["origin"].iloc[10]  # folds 0-9 calibrate
    offsets = calibrate(w, scored_from, taus=(0.5, 0.9))
    assert set(offsets["n_calib"]) == {10}
    calib = pd.Series(errors[:10], dtype=float)
    assert offsets.set_index("tau").loc[0.5, "offset"] == pytest.approx(
        calib.quantile(0.5)
    )
    assert offsets.set_index("tau").loc[0.9, "offset"] == pytest.approx(
        calib.quantile(0.9)
    )


def test_quantile_forecasts_add_the_offset_to_scored_weeks_only():
    errors = [float(e) for e in range(-10, 10)]
    w = weekly_totals(_predictions({"m": errors}))
    scored_from = w.sort_values("origin")["origin"].iloc[10]
    offsets = calibrate(w, scored_from, taus=(0.5,))
    qf = quantile_forecasts(w, offsets, scored_from)
    assert len(qf) == 10
    assert (qf["origin"] >= scored_from).all()
    np.testing.assert_allclose(qf["q"], qf["forecast"] + qf["offset"])


def test_expanding_calibration_uses_every_earlier_fold():
    errors = [float(e) for e in range(-10, 10)]
    w = weekly_totals(_predictions({"m": errors}))
    scored_from = w.sort_values("origin")["origin"].iloc[10]
    exp = calibrate_expanding(w, scored_from, taus=(0.5,))
    assert sorted(exp["fold"]) == list(range(10, 20))
    for _, row in exp.iterrows():
        past = np.array(errors[: int(row["fold"])])
        assert row["n_calib"] == len(past)
        assert row["offset"] == pytest.approx(np.quantile(past, 0.5))
    qf = quantile_forecasts(w, exp, scored_from)
    assert len(qf) == 10
    # The last scored fold's offset uses 19 earlier folds, the first only 10.
    assert qf.set_index("fold").loc[19, "n_calib"] == 19
    assert qf.set_index("fold").loc[10, "n_calib"] == 10


def test_score_quantiles_coverage_and_relative_pinball():
    errors = [float(e) for e in range(-10, 10)]
    w = weekly_totals(_predictions({"m": errors}))
    scored_from = w.sort_values("origin")["origin"].iloc[10]
    scored = score_quantiles(
        quantile_forecasts(w, calibrate(w, scored_from, taus=(0.5, 0.9)), scored_from)
    )
    np.testing.assert_array_equal(scored["covered"], scored["actual"] <= scored["q"])
    np.testing.assert_allclose(
        scored["pinball"], pinball(scored["actual"], scored["q"], scored["tau"])
    )
    level = w[w["origin"] >= scored_from]["actual"].mean()
    np.testing.assert_allclose(scored["pinball_rel"], scored["pinball"] / level)
    np.testing.assert_allclose(
        scored["shortfall"], np.maximum(scored["actual"] - scored["q"], 0)
    )
    np.testing.assert_allclose(
        scored["surplus"], np.maximum(scored["q"] - scored["actual"], 0)
    )


def test_calibration_on_a_known_distribution_gives_coverage_near_tau():
    rng = np.random.default_rng(0)
    errors = list(rng.normal(0, 30, 400))
    w = weekly_totals(_predictions({"m": errors}))
    scored_from = w.sort_values("origin")["origin"].iloc[200]
    scored = score_quantiles(
        quantile_forecasts(w, calibrate(w, scored_from), scored_from)
    )
    coverage = scored.groupby("tau")["covered"].mean()
    for tau, got in coverage.items():
        assert abs(got - tau) < 0.08, (tau, got)


def test_summary_and_ranking():
    w = weekly_totals(
        _predictions({"tight": [1.0, -1.0] * 10, "loose": [20.0, -20.0] * 10})
    )
    scored_from = w.loc[w["fold"] == 10, "origin"].iloc[0]
    scored = score_quantiles(
        quantile_forecasts(w, calibrate(w, scored_from), scored_from)
    )
    summary = summarise_quantiles(scored)
    assert set(summary["tau"]) == {0.3, 0.5, 0.7, 0.9}
    assert (summary["n_weeks"] == 10).all()
    ranking = ranking_by_tau(summary)
    assert list(ranking.index) == ["tight", "loose"]  # best at tau = 0.5 first
    assert (ranking.loc["tight"] < ranking.loc["loose"]).all()
    holiday = summarise_quantiles(scored, "holiday")
    assert (
        holiday["n_weeks"] == scored[scored["week_kind"] == "holiday"]["fold"].nunique()
    ).all()
