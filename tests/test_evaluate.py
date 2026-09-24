"""
Step 5 - the walk-forward harness and the scoring it feeds.

Documented behaviour under test (README, step5_evaluate docstrings):
  - one row per (series, fold, method, horizon day), actual + forecast + scale;
  - the folds tile the held-out window; nothing after the origin is visible;
  - RMSSE = RMSE / lag-7 naive RMSE on the pre-holdout training data, one
    denominator per series shared by every fold and method;
  - closure days are carried in `predictions` but dropped from every score;
  - a fold is a holiday fold if any scored day is in the holiday window;
  - bias = mean(forecast - actual);
  - the by-store table lists the busiest store first, best method first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.step1_problem import Config
from src.step4_models import BENCHMARKS
from src.step5_evaluate import (
    naive_scale,
    rmsse_by_store,
    run_walk_forward,
    score_folds,
    summarise,
    win_rates,
)
from tests.conftest import LAST, make_panel

BENCH = list(BENCHMARKS)


@pytest.fixture(scope="module")
def predictions(panel, tmp_path_factory):
    # Same layout as the `cfg` fixture, built once per module: the supervised
    # matrices are the slow part, and the benchmarks do not read them.
    cfg = Config(
        item_ids=("ITEM",),
        n_folds=4,
        min_train_days=200,
        cache_dir=tmp_path_factory.mktemp("cache"),
    )
    return run_walk_forward(panel, cfg, methods=BENCH, progress=False, use_cache=False)


# --------------------------------------------------------------------------- #
# naive_scale
# --------------------------------------------------------------------------- #


def test_naive_scale_is_the_rmse_of_a_lagged_naive_forecast():
    y = np.array([1.0, 2.0, 4.0, 7.0, 11.0, 16.0])
    # lag 1 differences: 1,2,3,4,5 -> RMS = sqrt(55/5)
    assert naive_scale(y, 1) == pytest.approx(np.sqrt(55 / 5))
    # lag 2 differences: 3,5,7,9 -> RMS = sqrt(164/4)
    assert naive_scale(y, 2) == pytest.approx(np.sqrt(164 / 4))


def test_naive_scale_edge_cases():
    assert np.isnan(naive_scale([1.0, 2.0], 2))
    assert naive_scale(np.full(20, 5.0), 7) == 0.0  # left as zero, RMSSE goes to inf


# --------------------------------------------------------------------------- #
# run_walk_forward
# --------------------------------------------------------------------------- #


def test_walk_forward_shape_and_tiling(predictions, panel, cfg):
    n_series = panel["id"].nunique()
    assert len(predictions) == n_series * cfg.n_folds * len(BENCH) * cfg.test_window
    assert set(predictions["method"]) == set(BENCH)
    assert (predictions["kind"] == "benchmark").all()
    last = panel["date"].max()
    scored = predictions.drop_duplicates("target_date")["target_date"].sort_values()
    # The scored days are exactly the held-out window, no gaps.
    expected = pd.date_range(cfg.holdout_start(last), last, freq="D")
    assert list(scored) == list(expected)
    assert (predictions["horizon"].between(1, cfg.test_window)).all()
    assert (
        (predictions["target_date"] - predictions["origin"]).dt.days
        == predictions["horizon"]
    ).all()


def test_walk_forward_actuals_match_the_panel(predictions, panel):
    s = panel.set_index(["id", "date"])["sales"].astype(float)
    sample = predictions.sample(50, random_state=0)
    want = s[list(zip(sample["id"], sample["target_date"], strict=True))].to_numpy()
    np.testing.assert_allclose(sample["actual"], want)


def test_naive_forecast_is_the_origin_days_sales(predictions, panel):
    s = panel.set_index(["id", "date"])["sales"].astype(float)
    naive = predictions[predictions["method"] == "naive"]
    want = s[list(zip(naive["id"], naive["origin"], strict=True))].to_numpy()
    np.testing.assert_allclose(naive["forecast"], want)


def test_seasonal_naive_forecast_is_last_weeks_same_day(predictions, panel):
    s = panel.set_index(["id", "date"])["sales"].astype(float)
    sn = predictions[predictions["method"] == "seasonal_naive"]
    want = s[
        list(zip(sn["id"], sn["target_date"] - pd.Timedelta(days=7), strict=True))
    ].to_numpy()
    np.testing.assert_allclose(sn["forecast"], want)


def test_scale_is_one_pre_holdout_denominator_per_series(predictions, panel, cfg):
    holdout = cfg.holdout_start(panel["date"].max())
    for sid, g in predictions.groupby("id"):
        train = panel[(panel["id"] == sid) & (panel["date"] < holdout)]["sales"]
        want = naive_scale(train, cfg.rmsse_scale_lag)
        assert g["scale"].nunique() == 1
        assert g["scale"].iloc[0] == pytest.approx(want)


def test_per_fold_scale_uses_each_folds_own_history(panel, tmp_path):
    cfg = Config(
        n_folds=3, min_train_days=200, rmsse_scale_window="per_fold", cache_dir=tmp_path
    )
    p = run_walk_forward(panel, cfg, methods=["naive"], progress=False, use_cache=False)
    for (sid, origin), g in p.groupby(["id", "origin"]):
        hist = panel[(panel["id"] == sid) & (panel["date"] <= origin)]["sales"]
        assert g["scale"].iloc[0] == pytest.approx(naive_scale(hist, cfg.rmsse_scale_lag))


def test_folds_without_enough_history_are_skipped(panel, tmp_path):
    n_days = panel["date"].nunique()
    cfg = Config(n_folds=4, min_train_days=n_days - 14, cache_dir=tmp_path)
    p = run_walk_forward(panel, cfg, methods=["naive"], progress=False, use_cache=False)
    # Only the folds whose origin has >= min_train_days of history survive.
    assert sorted(p["fold"].unique()) == [2, 3]


def test_unknown_method_is_rejected(panel, cfg):
    with pytest.raises(ValueError, match="unknown method"):
        run_walk_forward(panel, cfg, methods=["oracle"], use_cache=False)


def test_week_kind_is_holiday_when_any_scored_day_touches_the_window(cfg):
    origins = cfg.fold_origins(LAST)
    # Put a major event on the 3rd day of fold 1: fold 1 is a holiday fold,
    # every other fold is normal.
    event = origins[1] + pd.Timedelta(days=3)
    panel = make_panel(holidays=(event.strftime("%Y-%m-%d"),))
    p = run_walk_forward(panel, cfg, methods=["naive"], progress=False, use_cache=False)
    kinds = p.drop_duplicates(["id", "fold"]).set_index("fold")["week_kind"]
    assert (kinds.loc[1] == "holiday").all()
    assert (kinds.loc[[0, 2, 3]] == "normal").all()
    # The window is two days before to one day after - and that is what the
    # per-day flag says too.
    flagged = p[p["holiday_window"]]["target_date"].unique()
    assert sorted(flagged) == list(
        pd.date_range(event - pd.Timedelta(days=2), event + pd.Timedelta(days=1))
    )


def test_a_window_straddling_two_folds_marks_both_as_holiday(cfg):
    origins = cfg.fold_origins(LAST)
    # Event on the first scored day of fold 2: its two-day run-up is the end
    # of fold 1.
    event = origins[2] + pd.Timedelta(days=1)
    panel = make_panel(holidays=(event.strftime("%Y-%m-%d"),))
    p = run_walk_forward(panel, cfg, methods=["naive"], progress=False, use_cache=False)
    kinds = p.drop_duplicates(["id", "fold"]).set_index("fold")["week_kind"]
    assert (kinds.loc[[1, 2]] == "holiday").all()
    assert (kinds.loc[[0, 3]] == "normal").all()


def test_closure_days_are_carried_but_flagged(cfg):
    origins = cfg.fold_origins(LAST)
    closed = origins[0] + pd.Timedelta(days=4)
    panel = make_panel(closures=(closed.strftime("%Y-%m-%d"),))
    p = run_walk_forward(panel, cfg, methods=["naive"], progress=False, use_cache=False)
    assert set(p.loc[p["closure"], "target_date"]) == {closed}


def test_cache_round_trip_and_invalidation(panel, cfg):
    first = run_walk_forward(panel, cfg, methods=["naive"], progress=False)
    again = run_walk_forward(panel, cfg, methods=["naive"], progress=False)
    pd.testing.assert_frame_equal(first, again)
    # A different config must not be served the same file.
    other = Config(**{**cfg.__dict__, "n_folds": cfg.n_folds - 1})
    fewer = run_walk_forward(panel, other, methods=["naive"], progress=False)
    assert fewer["fold"].nunique() == cfg.n_folds - 1


def test_forecasters_see_no_future_sales(panel, cfg, monkeypatch):
    """The Context handed to a forecaster ends at the origin and carries no targets."""
    import src.step5_evaluate as s5

    seen = []

    def spy(ctx):
        seen.append(ctx)
        return np.zeros(ctx.horizon)

    monkeypatch.setitem(s5.ALL_FORECASTERS, "spy", spy)
    # in-process, so the spy records what the workers would see
    run_walk_forward(panel, cfg, methods=["spy"], progress=False, use_cache=False, n_jobs=1)
    assert seen
    for ctx in seen:
        assert ctx.history["date"].max() == ctx.origin
        assert ctx.targets["date"].min() > ctx.origin
        assert "sales" not in ctx.targets.columns


# --------------------------------------------------------------------------- #
# score_folds
# --------------------------------------------------------------------------- #


def _predictions_frame(rows):
    base = {
        "id": "S",
        "item_id": "I",
        "store_id": "S",
        "fold": 0,
        "origin": pd.Timestamp("2015-01-04"),
        "week_kind": "normal",
        "method": "m",
        "kind": "model",
        "scale": 2.0,
        "closure": False,
        "holiday_window": False,
    }
    out = []
    for h, r in enumerate(rows, 1):
        out.append(
            {
                **base,
                "horizon": h,
                "target_date": base["origin"] + pd.Timedelta(days=h),
                **r,
            }
        )
    return pd.DataFrame(out)


def test_score_folds_formulas():
    p = _predictions_frame(
        [
            {"actual": 10.0, "forecast": 12.0},  # err +2
            {"actual": 10.0, "forecast": 7.0},  # err -3
            {"actual": 10.0, "forecast": 10.0},  # err 0
        ]
    )
    s = score_folds(p).iloc[0]
    assert s["rmse"] == pytest.approx(np.sqrt((4 + 9 + 0) / 3))
    assert s["mae"] == pytest.approx(5 / 3)
    assert s["bias"] == pytest.approx(-1 / 3)  # mean(forecast - actual)
    assert s["rmsse"] == pytest.approx(s["rmse"] / 2.0)
    assert s["n_days"] == 3


def test_score_folds_drops_closure_days():
    p = _predictions_frame(
        [
            {"actual": 10.0, "forecast": 12.0},
            {"actual": 0.0, "forecast": 12.0, "closure": True},  # a shut store
        ]
    )
    s = score_folds(p).iloc[0]
    assert s["n_days"] == 1
    assert s["rmse"] == pytest.approx(2.0)


def test_zero_scale_gives_infinite_rmsse_rather_than_a_silent_number():
    p = _predictions_frame([{"actual": 10.0, "forecast": 11.0, "scale": 0.0}])
    assert np.isinf(score_folds(p).iloc[0]["rmsse"])


def test_rmsse_never_changes_who_wins_a_fold(predictions):
    scores = score_folds(predictions)
    for _, g in scores.groupby(["id", "fold"]):
        assert g["rmsse"].idxmin() == g["rmse"].idxmin()


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #


def test_summarise_splits_normal_and_holiday_and_sorts_best_first(cfg):
    origins = cfg.fold_origins(LAST)
    event = origins[1] + pd.Timedelta(days=3)
    panel = make_panel(holidays=(event.strftime("%Y-%m-%d"),))
    p = run_walk_forward(panel, cfg, methods=BENCH, progress=False, use_cache=False)
    scores = score_folds(p)
    table = summarise(scores)

    assert list(table["rmsse_mean"]) == sorted(table["rmsse_mean"])
    assert set(table["method"]) == set(BENCH)
    n_series = panel["id"].nunique()
    assert (table["n_folds"] == n_series * cfg.n_folds).all()
    assert (table["n_holiday"] == n_series).all()  # one holiday fold per store
    assert (table["n_normal"] == n_series * (cfg.n_folds - 1)).all()
    # The pooled mean is the fold-weighted mean of the two columns.
    pooled = (
        table["rmsse_normal"] * table["n_normal"]
        + table["rmsse_holiday"] * table["n_holiday"]
    ) / table["n_folds"]
    np.testing.assert_allclose(pooled, table["rmsse_mean"])
    # Bias column is the mean of per-fold bias, sign = forecast - actual.
    for _, row in table.iterrows():
        want = scores.loc[scores["method"] == row["method"], "bias"].mean()
        assert row["bias_mean"] == pytest.approx(want)


def test_win_rates_are_share_of_folds_with_lower_rmsse(predictions):
    scores = score_folds(predictions)
    table = win_rates(scores)
    assert set(table.columns) == set(BENCH)
    assert set(table.index) == set(BENCH)
    for bench in BENCH:
        assert np.isnan(table.loc[bench, bench])  # a benchmark against itself is blank
    wide = scores.pivot_table(index=["id", "fold"], columns="method", values="rmsse")
    for m in BENCH:
        for b in BENCH:
            if m != b:
                assert table.loc[m, b] == pytest.approx((wide[m] < wide[b]).mean())


def test_week_kind_filter_rejects_typos(predictions):
    scores = score_folds(predictions)
    with pytest.raises(ValueError):
        rmsse_by_store(scores, "holidays")
    with pytest.raises(ValueError):
        win_rates(scores, "Normal")


def test_rmsse_by_store_values_and_method_order(predictions):
    scores = score_folds(predictions)
    table = rmsse_by_store(scores)
    for store in table.index:
        for method in table.columns:
            want = scores[(scores["store_id"] == store) & (scores["method"] == method)][
                "rmsse"
            ].mean()
            assert table.loc[store, method] == pytest.approx(want)
    # Methods across, best (lowest mean RMSSE) first.
    means = table.mean(axis=0)
    assert list(means) == sorted(means)


def test_rmsse_by_store_lists_the_busiest_store_first(tmp_path):
    """
    BUG: the docstring, the `__main__` printout and the notebook all say
    "stores down, busiest first". The implementation sorts stores by their
    mean RMSE *ascending*, which puts the quietest store first - the exact
    reverse of the volume gradient the notebook asks the reader to read down.
    """
    panel = make_panel()
    cfg = Config(n_folds=4, min_train_days=200, cache_dir=tmp_path)
    p = run_walk_forward(
        panel, cfg, methods=["naive", "mean"], progress=False, use_cache=False
    )
    scores = score_folds(p)
    volume = panel.groupby("store_id")["sales"].mean().sort_values(ascending=False)
    assert list(volume.index) == ["S_BIG", "S_MID", "S_SMALL"]
    assert list(rmsse_by_store(scores).index) == ["S_BIG", "S_MID", "S_SMALL"]


def test_predictions_cache_is_per_method_and_keyed_on_that_methods_code(
    tmp_path, monkeypatch
):
    """
    Editing one model's module must invalidate that method's cache only. The
    digest is simulated: after the first run, the digest for "naive" is made
    to change and the run is repeated with both methods requested - "mean"
    must be served from cache and "naive" recomputed.
    """
    import src.step5_evaluate as s5

    panel = make_panel()
    cfg = Config(n_folds=3, min_train_days=200, cache_dir=tmp_path)
    first = run_walk_forward(panel, cfg, methods=["mean", "naive"], progress=False)
    files = sorted(p.name for p in (tmp_path / "predictions").glob("*.parquet"))
    assert (
        len(files) == 2 and files[0].startswith("mean_") and files[1].startswith("naive_")
    )

    real = s5._method_code

    def bumped(method):
        return "edited" if method == "naive" else real(method)

    monkeypatch.setattr(s5, "_method_code", bumped)
    calls = []
    orig = s5.ALL_FORECASTERS["mean"]
    s5.ALL_FORECASTERS["mean"] = lambda ctx: calls.append(1) or orig(ctx)
    try:
        second = run_walk_forward(panel, cfg, methods=["mean", "naive"], progress=False)
    finally:
        s5.ALL_FORECASTERS["mean"] = orig
    assert calls == [], "mean was recomputed although its code did not change"
    assert len(second) == len(first)
    assert len(list((tmp_path / "predictions").glob("naive_*.parquet"))) == 2


def test_predictions_cache_is_adopted_by_a_config_with_a_new_default_field(tmp_path):
    """
    A run made before a config field existed is reusable by a config that
    has the field at its default. Simulated by removing a field from the
    sidecar and asking for the same run again.
    """
    import json

    import src.step5_evaluate as s5

    panel = make_panel()
    cfg = Config(n_folds=3, min_train_days=200, cache_dir=tmp_path)
    run_walk_forward(panel, cfg, methods=["mean"], progress=False)
    meta = next((tmp_path / "predictions").glob("mean_*.json"))
    info = json.loads(meta.read_text())
    del info["config"]["mask_holidays"]  # pretend the run predates that field
    meta.write_text(json.dumps(info))
    parquet = meta.with_suffix(".parquet")
    parquet.rename(parquet.with_name("mean_oldkey.parquet"))
    meta.rename(meta.with_name("mean_oldkey.json"))

    assert s5._adopt_cached(cfg, "mean") is not None
    assert (
        s5._adopt_cached(
            Config(n_folds=3, min_train_days=200, cache_dir=tmp_path, mask_holidays=True),
            "mean",
        )
        is None
    )
