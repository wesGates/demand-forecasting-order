"""
Features: every value for a target day must be computable at the origin T.

Expected values are recomputed by date filtering, independently of the
positional slicing in `features.py`, as the validator's check 2 describes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    DOW_WINDOWS,
    LAGS,
    MIN_HISTORY,
    ROLL_WINDOWS,
    assert_no_leakage,
    build_fold_features,
    build_supervised,
    holiday_window,
    supervised_feature_columns,
)
from tests.conftest import make_panel


@pytest.fixture
def series():
    p = make_panel(n_days=300, stores={"S1": ("CA", 50.0)}, noise_sd=5.0, seed=1)
    return p.sort_values("date").reset_index(drop=True)


def _split(series, origin, horizon=7):
    history = series[series["date"] <= origin]
    targets = series[
        (series["date"] > origin)
        & (series["date"] <= origin + pd.Timedelta(days=horizon))
    ]
    return history, targets


def test_lags_rolls_and_dow_means_recomputed_by_date(series):
    origin = series["date"].iloc[-30]
    history, targets = _split(series, origin)
    feats = build_fold_features(history, targets, origin)
    s = series.set_index("date")["sales"].astype("float64")

    for k in LAGS:
        # lag_1 is the origin day itself; lag_k is k-1 days before it.
        assert feats[f"lag_{k}"].iloc[0] == pytest.approx(
            s[origin - pd.Timedelta(days=k - 1)]
        )
    for w in ROLL_WINDOWS:
        window = s[(s.index > origin - pd.Timedelta(days=w)) & (s.index <= origin)]
        assert feats[f"roll_mean_{w}"].iloc[0] == pytest.approx(window.mean())
        assert feats[f"roll_std_{w}"].iloc[0] == pytest.approx(window.std(ddof=1))
    hist = s[s.index <= origin]
    for k in DOW_WINDOWS:
        for row, day in enumerate(targets["date"]):
            same = hist[hist.index.dayofweek == day.dayofweek].tail(k)
            assert feats[f"dow_mean_{k}"].iloc[row] == pytest.approx(same.mean())
    assert list(feats["horizon"]) == list(range(1, 8))


def test_lag_and_rolling_columns_are_constant_across_the_fold(series):
    origin = series["date"].iloc[-30]
    history, targets = _split(series, origin)
    feats = build_fold_features(history, targets, origin)
    for col in [f"lag_{k}" for k in LAGS] + [f"roll_mean_{w}" for w in ROLL_WINDOWS]:
        assert feats[col].nunique() == 1


def test_history_past_origin_is_refused(series):
    origin = series["date"].iloc[-30]
    history, targets = _split(series, origin)
    too_long = series[series["date"] <= origin + pd.Timedelta(days=1)]
    with pytest.raises(ValueError, match="past"):
        build_fold_features(too_long, targets, origin)


def test_assert_no_leakage_refuses_both_directions(series):
    origin = series["date"].iloc[-30]
    history, targets = _split(series, origin)
    assert_no_leakage(history, targets, origin)  # clean split passes
    with pytest.raises(ValueError, match="LEAK"):
        assert_no_leakage(
            series[series["date"] <= origin + pd.Timedelta(days=1)], targets, origin
        )
    with pytest.raises(ValueError, match="LEAK"):
        assert_no_leakage(history, series[series["date"] <= origin].tail(3), origin)


def test_target_at_or_before_origin_is_refused(series):
    """
    Regression test for a fixed defect: the module docstring says leakage "is checked mechanically" by
    `assert_no_leakage`, but nothing calls it. `build_fold_features` only
    checks the history side; a target day at or before the origin - a
    horizon of 0 or less - is silently accepted and features are built for it.
    """
    origin = series["date"].iloc[-30]
    history, _ = _split(series, origin)
    bad_targets = series[series["date"] <= origin].tail(3)  # inside the training period
    with pytest.raises(ValueError):
        build_fold_features(history, bad_targets, origin)


def test_holiday_window_is_two_days_before_to_one_day_after():
    frame = pd.DataFrame(
        {
            "days_to_holiday": [3, 2, 1, 0, 30, 30],
            "days_since_holiday": [30, 30, 30, 0, 1, 2],
        }
    )
    assert list(holiday_window(frame)) == [False, True, True, True, True, False]


def test_masked_summaries_skip_the_holiday_window():
    p = make_panel(n_days=120, stores={"S1": ("CA", 50.0)}, holidays=("2013-04-25",))
    p = p.sort_values("date").reset_index(drop=True)
    # Put a spike on the holiday so masking is visible.
    p.loc[p["date"] == "2013-04-25", "sales"] = 1000.0
    origin = pd.Timestamp("2013-04-26")  # the day after: inside the window
    history, targets = _split(p, origin)
    masked = build_fold_features(history, targets, origin, mask_holidays=True)
    plain = build_fold_features(history, targets, origin, mask_holidays=False)
    assert plain["roll_mean_7"].iloc[0] > masked["roll_mean_7"].iloc[0]
    assert masked["roll_mean_7"].iloc[0] < 100
    # Lags are left alone - each names one specific day.
    assert masked["lag_2"].iloc[0] == plain["lag_2"].iloc[0] == 1000.0


def test_supervised_matrix_rows_are_leak_free_by_date(series):
    sup = build_supervised(series, horizon=7)
    assert (sup["target_date"] > sup["origin_date"]).all()
    assert ((sup["target_date"] - sup["origin_date"]).dt.days == sup["horizon"]).all()
    # First origin has MIN_HISTORY days before it, so every feature is filled.
    first = sup[sup["origin_date"] == sup["origin_date"].min()]
    assert first["origin_date"].iloc[0] == series["date"].iloc[MIN_HISTORY]
    assert not first[[f"lag_{k}" for k in LAGS]].isna().any().any()
    assert not first[[f"dow_mean_{k}" for k in DOW_WINDOWS]].isna().any().any()
    # The last origin's targets end on the final day of the series.
    assert sup["target_date"].max() == series["date"].max()
    # Every origin contributes exactly `horizon` rows.
    assert (sup.groupby("origin_date").size() == 7).all()


def test_supervised_target_matches_the_panel(series):
    sup = build_supervised(series, horizon=7)
    s = series.set_index("date")["sales"].astype(float)
    sample = sup.sample(20, random_state=0)
    np.testing.assert_allclose(sample["target"], s[sample["target_date"]].to_numpy())


def test_feature_columns_exclude_bookkeeping_and_include_identity_only_when_pooled(
    series,
):
    sup = build_supervised(series, horizon=7)
    cols = supervised_feature_columns(sup)
    for banned in (
        "id",
        "store_id",
        "item_id",
        "target",
        "target_date",
        "origin_date",
        "sales",
    ):
        assert banned not in cols
    assert "horizon" in cols
    pooled = supervised_feature_columns(sup, pool_by="item_id")
    assert "store_id" in pooled and "item_id" in pooled
    assert "sell_price" not in cols
    assert "sell_price" in supervised_feature_columns(
        build_supervised(series, 7, use_price=True)
    )
