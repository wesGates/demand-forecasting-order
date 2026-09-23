"""
The validator. Run it before trusting any number this project produces.

    python -m src.validate

Nine checks in seven groups, each aimed at a different way of being wrong. What
they have in common is the only thing that matters: **every bug they catch
still produces plausible-looking output.** A result that is obviously broken
gets fixed the moment you see it; a result that is quietly wrong ends up in a
report.

  1. Closed-form benchmarks - does each method compute what its name says?
     On a series that is 10 every day, every method must return exactly 10. On
     a straight ramp, drift must extrapolate it perfectly. Catches off-by-one
     indexing, which on real data looks entirely reasonable.

  2. Feature spot-checks - is `lag_7` really seven days back?
     Each feature is recomputed from the raw panel by *date filtering*, while
     the implementation uses array slicing. Two different routes to the same
     number; if they disagree, one of them is wrong.

  3. SNAP flags - does each row carry its own state's schedule?
     Every store-date compared against the calendar's column for that state.
     Reading the wrong state's column would still give a plausible 0/1 flag
     on a third of days; only a date-by-date comparison catches it.

  4. Noise floor and shuffled target - is anything reading the future?
     Synthetic data whose irreducible error is known by construction, and a
     scrambled target with no pattern left to find. The structural date
     assertion in `features.py` is stronger for the leak we know about; these
     are the net for the ones we do not.

  5. Determinism - does the same input give the same answer twice?
     Cheap, and it is what makes a reported number reproducible.

  6. Holiday calendar - are the proximity counts and closure flags right?
     Recounted by hand from the raw calendar for every date in the panel.

  7. Quantile scoring - does the order-quantity arithmetic do what it says?
     Pinball closed form, and coverage on errors of known distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import (
    DOW_WINDOWS,
    LAGS,
    ROLL_WINDOWS,
    build_fold_features,
    build_supervised,
    holiday_window,
)
from src.step1_problem import STUDY_ITEMS, Config
from src.step2_data import (
    CLOSURE_EVENTS,
    HOLIDAY_CLIP_DAYS,
    MAJOR_EVENTS,
    SNAP_BY_STATE,
    load_panel,
)
from src.step4_models import BENCHMARKS, Context, fit_predict_xgboost, reset_run_state

TOL = 1e-9


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _series_frame(values: np.ndarray, start: str = "2013-01-07") -> pd.DataFrame:
    """A minimal panel for one synthetic series. Starts on a Monday."""
    dates = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame(
        {
            "id": "SYNTH",
            "store_id": "S1",
            "item_id": "SYNTH_ITEM",
            "date": dates,
            "sales": np.asarray(values, dtype=float),
            "snap": 0,
            "event_name_1": pd.Series([None] * len(values), dtype=object),
            "is_holiday": 0,
            "days_to_holiday": 30,
            "days_since_holiday": 30,
            "closure": False,
        }
    )


def _context(frame: pd.DataFrame, horizon: int = 7, **kw) -> Context:
    """Split a synthetic frame into history + targets at the natural origin."""
    history = frame.iloc[:-horizon]
    targets = frame.iloc[-horizon:].drop(columns=["sales"])
    return Context(
        history=history,
        targets=targets,
        origin=history["date"].iloc[-1],
        horizon=horizon,
        series_id="SYNTH",
        **kw,
    )


def _synthetic_panel(
    n_series: int = 6, n_days: int = 1000, noise_sd: float = 2.0, seed: int = 0
) -> pd.DataFrame:
    """
    A panel whose irreducible error is known exactly.

    sales = level + weekly shape + slow trend + noise(0, noise_sd)

    Everything but the noise is a deterministic function of the date, so a
    perfect forecaster would predict it exactly and be left with the noise
    alone. `noise_sd` is therefore the best RMSE physically achievable.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2013-01-07", periods=n_days, freq="D")
    frames = []
    for k in range(n_series):
        weekly = 4.0 * np.sin(2 * np.pi * dates.dayofweek / 7)
        signal = (20 + 6 * k) + weekly + np.linspace(0, 3, n_days)
        frames.append(
            pd.DataFrame(
                {
                    "id": f"SYNTH_{k:02d}",
                    "store_id": f"S{k:02d}",
                    "item_id": "SYNTH_ITEM",
                    "date": dates,
                    "sales": np.maximum(signal + rng.normal(0, noise_sd, n_days), 0),
                    "snap": 0,
                    "event_name_1": pd.Series([None] * n_days, dtype=object),
                    "is_holiday": 0,
                    "days_to_holiday": 30,
                    "days_since_holiday": 30,
                    "closure": False,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _walk_forward_rmse(panel: pd.DataFrame, cfg: Config) -> tuple[float, float, int]:
    """
    Run XGBoost and the mean benchmark over every fold of a synthetic panel.

    Returns (model RMSE, mean-benchmark RMSE, number of fold-series compared).
    Used by both the noise-floor and shuffled-target checks.
    """
    origins = cfg.fold_origins(panel["date"].max())
    model_err, bench_err, n = [], [], 0

    # Honour the pooling setting: with `pool_by` every series in the pool
    # trains on all of their rows, exactly as the harness does.
    matrices = {
        sid: build_supervised(s.sort_values("date").reset_index(drop=True), cfg.horizon)
        for sid, s in panel.groupby("id", sort=True, observed=True)
    }
    reset_run_state()

    for sid, series in panel.groupby("id", sort=True, observed=True):
        series = series.sort_values("date").reset_index(drop=True)
        if cfg.pool_by is None:
            pool = matrices[sid]
        else:
            key = series[cfg.pool_by].iloc[0]
            members = panel.loc[panel[cfg.pool_by] == key, "id"].unique()
            pool = pd.concat([matrices[m] for m in members], ignore_index=True)

        for origin in origins:
            history = series[series["date"] <= origin]
            targets = series[
                (series["date"] > origin)
                & (series["date"] <= origin + pd.Timedelta(days=cfg.horizon))
            ]
            if len(targets) < cfg.horizon or len(history) < cfg.min_train_days:
                continue

            actual = targets["sales"].to_numpy(dtype=float)
            ctx = Context(
                history=history,
                targets=targets.drop(columns=["sales"]),
                origin=origin,
                horizon=cfg.horizon,
                series_id=sid,
                season=cfg.season,
                train_pool=pool,
                pool_by=cfg.pool_by,
                seed=cfg.seed,
            )
            pred = fit_predict_xgboost(ctx)
            if np.isnan(pred).any():
                continue

            model_err.append((actual - pred) ** 2)
            bench_err.append((actual - BENCHMARKS["mean"](ctx)) ** 2)
            n += 1

    return (
        float(np.sqrt(np.concatenate(model_err).mean())),
        float(np.sqrt(np.concatenate(bench_err).mean())),
        n,
    )


# --------------------------------------------------------------------------- #
# 1. Closed-form benchmarks
# --------------------------------------------------------------------------- #


def check_benchmarks_closed_form() -> dict:
    """Every benchmark, on series whose correct answer is known by hand."""
    failures = []

    # A flat series: there is only one defensible forecast, and it is 10.
    ctx = _context(_series_frame(np.full(200, 10.0)))
    for name, fn in BENCHMARKS.items():
        got = fn(ctx)
        if not np.allclose(got, 10.0, atol=TOL):
            failures.append(f"{name} on a constant-10 series returned {got}")

    # A ramp y_t = t. Slope is exactly 1 per day, so drift must continue it.
    n = 200
    ctx = _context(_series_frame(np.arange(1, n + 1, dtype=float)))
    last = float(n - 7)  # history stops 7 days before the end
    expected = {
        "naive": np.full(7, last),
        "drift": last + np.arange(1, 8),
        "mean": np.full(7, np.arange(1, n - 6).mean()),
        "seasonal_naive": np.arange(last - 6, last + 1),
        # history is only 193 days, shorter than 364, so this must fall back to
        # the weekly seasonal naive
        "seasonal_naive_364": np.arange(last - 6, last + 1),
        "moving_average_28": np.full(7, np.arange(last - 27, last + 1).mean()),
    }
    # And with 400 days of history the lag-364 benchmark must reach back a year.
    n2 = 400
    ctx2 = _context(_series_frame(np.arange(1, n2 + 1, dtype=float)))
    want = np.arange(n2 - 7 + 1, n2 + 1) - 364.0  # target day t -> value at t-364
    got = BENCHMARKS["seasonal_naive_364"](ctx2)
    if not np.allclose(got, want, atol=1e-6):
        failures.append(
            f"seasonal_naive_364 on a 400-day ramp returned {got}, expected {want}"
        )
    for name, want in expected.items():
        got = BENCHMARKS[name](ctx)
        if not np.allclose(got, want, atol=1e-6):
            failures.append(f"{name} on a ramp returned {got}, expected {want}")

    return {
        "check": "closed-form benchmarks",
        "tested": f"{len(BENCHMARKS)} methods on 2 series with known answers",
        "failures": failures,
        "verdict": "ok" if not failures else "FAILED",
    }


# --------------------------------------------------------------------------- #
# 2. Feature spot-checks
# --------------------------------------------------------------------------- #


def check_features_hand_computed(cfg: Config | None = None) -> dict:
    """
    Recompute every feature from the raw panel by date, and compare.

    The implementation slices numpy arrays by position; this recomputes the same
    quantities by filtering the DataFrame on dates. Two independent routes - a
    single off-by-one shows up as a mismatch instead of as a plausible number.
    """
    cfg = cfg or Config(item_ids=STUDY_ITEMS)
    panel = load_panel(cfg, verbose=False)
    series = panel[panel["id"] == panel["id"].iloc[0]].sort_values("date")

    failures, checked = [], 0
    for offset, mask in (
        (7, False),
        (200, False),
        (900, False),
        (150, True),
        (520, True),
    ):
        # several origins, not just the convenient one; two with holiday
        # masking on, so that code path is recomputed by date as well
        origin = series["date"].max() - pd.Timedelta(days=offset)
        history = series[series["date"] <= origin]
        targets = series[
            (series["date"] > origin)
            & (series["date"] <= origin + pd.Timedelta(days=cfg.horizon))
        ]
        feats = build_fold_features(history, targets, origin, mask_holidays=mask)
        # Days the masked summaries may use: outside the holiday window.
        usable = history[~holiday_window(history)] if mask else history

        # `sales` is stored as float32 to halve panel memory. The feature code
        # promotes to float64 before doing any arithmetic; this recomputation
        # must do the same, or the two sides differ at the 1e-6 level purely
        # from float32 accumulation and every rolling mean looks like a bug.
        def sales_on(day: pd.Timestamp) -> float:
            return float(series.loc[series["date"] == day, "sales"].iloc[0])

        def f64(values) -> pd.Series:
            return values.astype("float64")

        for k in LAGS:
            # lag_k counts back from the origin, so lag_1 IS the origin day.
            want = sales_on(origin - pd.Timedelta(days=k - 1))
            got = float(feats[f"lag_{k}"].iloc[0])
            checked += 1
            if abs(got - want) > TOL:
                failures.append(f"lag_{k} at {origin.date()}: got {got}, want {want}")

        for w in ROLL_WINDOWS:
            window = f64(usable[usable["date"] > origin - pd.Timedelta(days=w)]["sales"])
            for stat, want in (("mean", window.mean()), ("std", window.std(ddof=1))):
                got = float(feats[f"roll_{stat}_{w}"].iloc[0])
                checked += 1
                if not np.isclose(got, float(want), rtol=1e-9, atol=1e-9):
                    failures.append(
                        f"roll_{stat}_{w} at {origin.date()}: got {got}, want {want}"
                    )

        for k in DOW_WINDOWS:
            for row in range(len(targets)):
                dow = targets["date"].iloc[row].dayofweek
                same = f64(usable[usable["date"].dt.dayofweek == dow]["sales"])
                want = float(same.tail(k).mean())
                got = float(feats[f"dow_mean_{k}"].iloc[row])
                checked += 1
                if not np.isclose(got, want, rtol=1e-9, atol=1e-9):
                    failures.append(
                        f"dow_mean_{k} h={row + 1} at {origin.date()}: "
                        f"got {got}, want {want}"
                    )

        # horizon must be the real day gap, 1..h
        if list(feats["horizon"]) != list(range(1, len(targets) + 1)):
            failures.append(f"horizon at {origin.date()}: {list(feats['horizon'])}")
        checked += 1

    return {
        "check": "feature spot-checks",
        "tested": f"{checked} feature values against date-filtered recomputation",
        "failures": failures[:8],
        "verdict": "ok" if not failures else "FAILED",
    }


# --------------------------------------------------------------------------- #
# 3. Noise floor and shuffled target
# --------------------------------------------------------------------------- #


def check_layouts_and_pooling(noise_sd: float = 2.0, seed: int = 7) -> dict:
    """
    The noise-floor and shuffled-target nets, repeated under the two
    settings the default checks do not exercise: every day an origin
    (`fold_step=1`, overlapping windows) and one model pooled across the
    series. A leak that only opens when windows overlap, or when other
    series' rows sit in the training pool, would show here and nowhere else.
    """
    panel = _synthetic_panel(n_series=6, n_days=1000, noise_sd=noise_sd)
    rng = np.random.default_rng(seed)
    shuffled = panel.copy()
    shuffled["sales"] = rng.permutation(shuffled["sales"].to_numpy())
    layouts = {
        "every-day origins": Config(n_folds=30, fold_step=1, min_train_days=200),
        "pooled": Config(n_folds=10, min_train_days=200, pool_by="item_id"),
    }
    failures, detail = [], {}
    for name, cfg in layouts.items():
        floor_model, _, n = _walk_forward_rmse(panel, cfg)
        model, bench, _ = _walk_forward_rmse(shuffled, cfg)
        detail[name] = {
            "noise_floor_ratio": round(floor_model / noise_sd, 3),
            "shuffled_ratio": round(model / bench, 3),
            "folds": n,
        }
        if floor_model < noise_sd:
            failures.append(
                f"{name}: beat the noise floor ({floor_model:.3f} < {noise_sd})"
            )
        if model / bench < 0.97:
            failures.append(
                f"{name}: learned from a shuffled target (ratio {model / bench:.3f})"
            )
    return {
        "check": "layouts and pooling",
        "tested": "noise floor and shuffled target under fold_step=1 and pool_by",
        **detail,
        "failures": failures,
        "verdict": "ok" if not failures else "FAILED",
    }


def check_noise_floor(noise_sd: float = 2.0) -> dict:
    """
    Model error must not fall below the noise that was injected.

    No honest forecaster can beat noise you generated yourself. Error
    meaningfully under the floor means information is arriving from the future.
    """
    cfg = Config(n_folds=10, min_train_days=200)
    observed, _, n = _walk_forward_rmse(_synthetic_panel(noise_sd=noise_sd), cfg)
    ratio = observed / noise_sd

    # ~420 scored points, so the RMSE's own sampling error is a few percent.
    # 0.90 is roughly three standard errors below the floor.
    return {
        "check": "noise floor",
        "tested": f"{n} fold-series windows on synthetic data",
        "floor_rmse": round(noise_sd, 3),
        "observed_rmse": round(observed, 3),
        "ratio": round(ratio, 3),
        "verdict": "LEAK SUSPECTED" if ratio < 0.90 else "ok",
    }


def check_shuffled_target(seed: int = 7) -> dict:
    """
    With the target scrambled, no model may beat a mean-predicting benchmark.

    The benchmark must be mean-based. Against seasonal naive this test fires on
    a perfectly healthy pipeline: seasonal naive stakes everything on one past
    day, so on noise its error is about sqrt(2) worse than the mean, and any
    model that predicts near the mean beats it without skill.
    """
    rng = np.random.default_rng(seed)
    panel = _synthetic_panel(seed=seed)
    panel["sales"] = (
        panel.groupby("id", observed=True)["sales"]
        .transform(lambda s: rng.permutation(s.to_numpy()))
        .to_numpy()
    )

    cfg = Config(n_folds=10, min_train_days=200)
    model, bench, n = _walk_forward_rmse(panel, cfg)
    ratio = model / bench

    # Model should be no better than the mean - ratio at or above 1.0.
    return {
        "check": "shuffled target",
        "tested": f"{n} fold-series windows, pattern destroyed",
        "model_rmse": round(model, 3),
        "mean_benchmark_rmse": round(bench, 3),
        "ratio_model_over_benchmark": round(ratio, 3),
        "expected": "at or above 1.0 - nothing left to learn",
        "verdict": "LEAK SUSPECTED" if ratio < 0.95 else "ok",
    }


# --------------------------------------------------------------------------- #
# 4. Determinism
# --------------------------------------------------------------------------- #


def check_determinism(cfg: Config | None = None) -> dict:
    """Identical inputs must produce byte-identical forecasts."""
    cfg = cfg or Config(item_ids=STUDY_ITEMS)
    panel = load_panel(cfg, verbose=False)
    series = panel[panel["id"] == panel["id"].iloc[0]].sort_values("date")
    pool = build_supervised(series, cfg.horizon)
    origin = cfg.fold_origins(panel["date"].max())[-1]

    history = series[series["date"] <= origin]
    targets = series[
        (series["date"] > origin)
        & (series["date"] <= origin + pd.Timedelta(days=cfg.horizon))
    ]
    ctx = Context(
        history=history,
        targets=targets.drop(columns=["sales"]),
        origin=origin,
        horizon=cfg.horizon,
        series_id=series["id"].iloc[0],
        season=cfg.season,
        train_pool=pool,
        seed=cfg.seed,
    )
    a, b = fit_predict_xgboost(ctx), fit_predict_xgboost(ctx)

    return {
        "check": "determinism",
        "tested": "same config, two runs, xgboost",
        "identical": bool(np.array_equal(a, b)),
        "max_difference": float(np.max(np.abs(a - b))),
        "verdict": "ok" if np.array_equal(a, b) else "FAILED",
    }


# --------------------------------------------------------------------------- #
# 5. SNAP flags
# --------------------------------------------------------------------------- #


def check_snap_flags(cfg: Config | None = None) -> dict:
    """
    Every row's SNAP flag must equal the raw calendar's column for that row's
    own state, on every date.

    The calendar carries three columns - snap_CA, snap_TX, snap_WI - and the
    loader picks one per row. The failure this guards against is quiet: if
    every row read snap_CA, the flag would still be 0/1, still fire on about a
    third of days, and every downstream plot would look perfectly plausible.
    Only a date-by-date comparison against the source catches it.

    The days-of-month are reported too, so a human can eyeball that each state
    has its own schedule rather than all three sharing one.
    """
    cfg = cfg or Config(item_ids=STUDY_ITEMS)
    panel = load_panel(cfg, verbose=False)
    calendar = pd.read_csv(cfg.data_dir / "calendar.csv", parse_dates=["date"])

    failures, days_by_state, compared = [], {}, 0
    for state, column in SNAP_BY_STATE.items():
        rows = panel[panel["state_id"] == state].drop_duplicates("date")
        if rows.empty:
            continue
        got = rows.set_index("date")["snap"].astype(int)
        want = calendar.set_index("date")[column].reindex(got.index).astype(int)
        wrong = int((got != want).sum())
        compared += len(got)
        if wrong:
            failures.append(
                f"{state}: {wrong} of {len(got)} dates disagree with {column}"
            )
        days_by_state[state] = sorted(
            int(d) for d in rows.loc[rows["snap"] == 1, "date"].dt.day.unique()
        )

    return {
        "check": "SNAP flags per state",
        "tested": f"{compared:,} store-dates against calendar.csv, {len(days_by_state)} states",
        "snap_days_of_month": days_by_state,
        "failures": failures,
        "verdict": "ok" if not failures else "FAILED",
    }


# --------------------------------------------------------------------------- #
# 6. Holiday calendar and closure handling
# --------------------------------------------------------------------------- #


def check_holiday_calendar(cfg: Config | None = None) -> dict:
    """
    The proximity columns must agree with a by-hand recount from the raw
    calendar, and the closure flag must sit on exactly the closure events.

    Both are the kind of thing that fails plausibly: an off-by-one in
    `days_to_holiday` still gives small integers near holidays, and a closure
    flag on the wrong day still imputes *something*.
    """
    cfg = cfg or Config(item_ids=STUDY_ITEMS)
    panel = load_panel(cfg, verbose=False)
    calendar = pd.read_csv(cfg.data_dir / "calendar.csv", parse_dates=["date"])
    names = calendar[["event_name_1", "event_name_2"]]
    major = set(calendar.loc[names.isin(MAJOR_EVENTS).any(axis=1), "date"])
    closures = set(calendar.loc[names.isin(CLOSURE_EVENTS).any(axis=1), "date"])

    days = panel.drop_duplicates("date").set_index("date").sort_index()
    failures, checked = [], 0
    for day, row in days.iterrows():
        ahead = [(m - day).days for m in major if m >= day]
        behind = [(day - m).days for m in major if m <= day]
        want_to = min(ahead + [HOLIDAY_CLIP_DAYS]) if ahead else HOLIDAY_CLIP_DAYS
        want_since = min(behind + [HOLIDAY_CLIP_DAYS]) if behind else HOLIDAY_CLIP_DAYS
        want_to, want_since = (
            min(want_to, HOLIDAY_CLIP_DAYS),
            min(want_since, HOLIDAY_CLIP_DAYS),
        )
        got = (
            int(row["days_to_holiday"]),
            int(row["days_since_holiday"]),
            int(row["is_holiday"]),
        )
        want = (want_to, want_since, int(day in major))
        checked += 1
        if got != want:
            failures.append(f"{day.date()}: got to/since/is={got}, want {want}")

    flagged = set(panel.loc[panel["closure"], "date"])
    if flagged != (closures & set(days.index)):
        failures.append(f"closure flag on {sorted(d.date() for d in flagged)[:5]}...")
    # Imputed closure days must no longer be zero on a continuously stocked item.
    still_zero = int((panel.loc[panel["closure"], "sales"] == 0).sum())
    if still_zero:
        failures.append(f"{still_zero} closure rows still read zero after imputation")

    return {
        "check": "holiday calendar and closures",
        "tested": f"{checked:,} dates recounted from calendar.csv; {len(flagged)} closure dates",
        "major_events": sorted(MAJOR_EVENTS),
        "failures": failures[:8],
        "verdict": "ok" if not failures else "FAILED",
    }


# --------------------------------------------------------------------------- #
# 7. Quantile scoring
# --------------------------------------------------------------------------- #


def check_quantile_scoring(seed: int = 3) -> dict:
    """
    The order-quantity arithmetic, on inputs whose right answer is known.

    Pinball at tau = 0.5 must equal the absolute error (FPP's factor of two is
    what makes that true), and it must reward a correct quantile: on errors
    drawn from a known distribution, calibrating on one sample and scoring on
    another must give coverage within a few points of tau, and a deliberately
    shifted quantile must score worse.
    """
    from src.order import (
        calibrate,
        pinball,
        quantile_forecasts,
        score_quantiles,
        weekly_totals,
    )

    failures = []
    rng = np.random.default_rng(seed)

    # --- pinball closed form ------------------------------------------------
    y, q = rng.normal(0, 10, 500), rng.normal(0, 10, 500)
    if not np.allclose(pinball(y, q, 0.5), np.abs(y - q), atol=TOL):
        failures.append("pinball at tau=0.5 is not the absolute error")
    if not np.allclose(pinball([10.0], [0.0], 0.9), [18.0]) or not np.allclose(
        pinball([0.0], [10.0], 0.9), [2.0]
    ):
        failures.append(
            "pinball asymmetry wrong: shortfall of 10 at tau=0.9 should cost 18, surplus 2"
        )

    # --- calibration transfers between independent samples ------------------
    # Two years of weekly errors from one N(0, 30) distribution, as a fake
    # predictions table: one series, one method, seven identical days a week.
    n_weeks, horizon = 300, 7
    origins = pd.date_range("2013-01-06", periods=n_weeks, freq="7D")
    rows = []
    for k, origin in enumerate(origins):
        err = rng.normal(0, 30)
        for h in range(1, horizon + 1):
            rows.append(
                {
                    "id": "S",
                    "item_id": "I",
                    "store_id": "S",
                    "fold": k,
                    "origin": origin,
                    "week_kind": "normal",
                    "method": "m",
                    "kind": "model",
                    "horizon": h,
                    "target_date": origin + pd.Timedelta(days=h),
                    "actual": 100.0 + err / horizon,
                    "forecast": 100.0,
                    "scale": 1.0,
                    "closure": False,
                    "holiday_window": False,
                }
            )
    weekly = weekly_totals(pd.DataFrame(rows))
    scored_from = origins[n_weeks // 2]
    offsets = calibrate(weekly, scored_from, taus=(0.3, 0.5, 0.9))
    scored = score_quantiles(quantile_forecasts(weekly, offsets, scored_from))
    coverage = scored.groupby("tau")["covered"].mean()
    for tau, got in coverage.items():
        if abs(got - tau) > 0.08:
            failures.append(f"coverage at tau={tau}: {got:.2f}, expected within 0.08")
    # A shifted quantile must score worse than the calibrated one.
    s9 = scored[scored["tau"] == 0.9]
    shifted = pinball(s9["actual"], s9["q"] + 30, 0.9).mean()
    if not shifted > s9["pinball"].mean():
        failures.append(
            "an over-shifted 0.9 quantile did not score worse than the calibrated one"
        )

    return {
        "check": "quantile scoring",
        "tested": f"pinball closed form on 500 pairs; coverage on {n_weeks // 2} held-out weeks",
        "coverage": {float(t): round(float(c), 3) for t, c in coverage.items()},
        "failures": failures,
        "verdict": "ok" if not failures else "FAILED",
    }


# --------------------------------------------------------------------------- #


def run_all() -> bool:
    """Run every check. Returns True only if all pass."""
    checks = [
        check_benchmarks_closed_form,
        check_features_hand_computed,
        check_snap_flags,
        check_holiday_calendar,
        check_quantile_scoring,
        check_determinism,
        check_noise_floor,
        check_shuffled_target,
        check_layouts_and_pooling,
    ]

    ok = True
    for fn in checks:
        result = fn()
        name = result.pop("check")
        print(f"\n--- {name} ---")
        for key, value in result.items():
            if key == "failures" and not value:
                continue
            print(f"  {key}: {value}")
        if "ok" not in str(result.get("verdict", "")):
            ok = False

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILED - see above"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run_all() else 1)
