"""
Step 5 - Using and evaluating the forecasts (FPP §1.6, step 5).

This is the harness. It walks every series through every fold, asks every
forecaster for its forecast, and records what actually happened. Everything
reported afterwards - RMSSE tables, win rates, the plots - derives from the one
long table `run_walk_forward` returns, so there is exactly one place where a
forecast meets an actual.

**Walk-forward validation** (FPP §5.10, "time series cross-validation"). The
folds come from `Config.fold_origins`: each is a forecast origin T, and the
`test_window` days after it are scored. Nothing after T is visible to any
forecaster - `Context` carries no future sales, and the learned model's
training rows are filtered by `target_date <= T` in step 4. The folds tile the
held-out period end to end, so the scored days are exactly the ones step 3's
classification never saw.

**RMSSE** (FPP §5.8). Root mean squared *scaled* error: the forecast's RMSE
divided by the RMSE a naive forecast would have had on the training data.

    RMSSE = RMSE(forecast) / sqrt( mean( (y_t - y_{t-lag})^2 ) over training )

A value of 1.0 means "as good as naively repeating the value from `lag` days
ago"; 0.8 means 20% better. Because the denominator is a property of the series
and not of the forecast, RMSSE is comparable across stores of very different
size - a 5-unit error means something different at 100 units/day than at 15.

Two details are `Config` decisions, and the config comments say why:
`rmsse_scale_lag` (FPP says `season` for seasonal data; M5 used 1) and
`rmsse_scale_window` ("pre_holdout": one denominator per series over all
training data before the first scored day, shared by every fold and method).

A property worth having straight: within one series-fold every method shares
the same denominator, so RMSSE **never changes which method wins that fold** -
it is a rescaling of RMSE. What it changes is how you aggregate across stores.

RMSE, MAE and bias are kept alongside as the plain-units view. Bias is
`mean(forecast - actual)`: positive means over-forecasting, and it matters in
its own right for replenishment, where a consistent 5% under-forecast is worse
than noisy-but-centred.

**Two kinds of week.** Every score is also reported split into *normal* and
*holiday* folds. A fold is a holiday fold if any of its scored days falls in
the window from two days before a major event to one day after it - the
run-up and the hangover the event-effect table shows. The split is not there
to hide holidays; it is there because a single pooled number cannot say
whether a method's advantage comes from the fifty ordinary weeks or from the
handful where the calendar does the work. Both columns are reported, always.

**Closure days are not scored.** Step 2 flags the days the stores were shut
and imputes their sales so the following week's features are sane. Those days
are still in `predictions` (so a forecast plot shows them) but `score_folds`
drops them: forecasting a locked door is not a demand question.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src import step2_data
from src.features import FEATURE_VERSION, build_supervised, holiday_window
from src.step1_problem import Config
from src.step4_models import ALL_FORECASTERS, BENCHMARKS, Context, reset_run_state

# --------------------------------------------------------------------------- #
# Supervised matrices, cached
# --------------------------------------------------------------------------- #


def _supervised_path(cfg: Config, series_id: str):
    # The panel version is in the key too: a loader change (closure imputation
    # was one) changes the sales the lags and targets are built from, and a
    # matrix built from the old panel must not be served for the new one.
    key = repr(
        (
            FEATURE_VERSION,
            step2_data.CACHE_VERSION,  # via the module, so a bump is always seen
            series_id,
            cfg.horizon,
            cfg.use_price,
            cfg.mask_holidays,
        )
    )
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return cfg.cache_dir / f"supervised_v{FEATURE_VERSION}_{digest}.parquet"


def supervised_matrices(df: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    """
    One supervised matrix per series, built once and cached to parquet.

    Building takes ~8s per series; reading back takes well under a second. The
    cache key includes `FEATURE_VERSION`, so editing a feature and forgetting
    to rebuild cannot serve stale rows - the old file simply stops matching.
    """
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for series_id, series in df.groupby("id", sort=True, observed=True):
        path = _supervised_path(cfg, series_id)
        if path.exists():
            out[series_id] = pd.read_parquet(path)
        else:
            matrix = build_supervised(
                series,
                cfg.horizon,
                use_price=cfg.use_price,
                mask_holidays=cfg.mask_holidays,
            )
            matrix.to_parquet(path, index=False)
            out[series_id] = matrix
    return out


# --------------------------------------------------------------------------- #
# The RMSSE denominator
# --------------------------------------------------------------------------- #


def naive_scale(y, lag: int) -> float:
    """
    RMSE of a lag-`lag` naive forecast over the given values - the RMSSE
    denominator of FPP §5.8, in its squared-error form.

    Returns NaN if there is nothing to difference. A series that never changes
    has a zero denominator; that is left as 0 so the resulting RMSSE is `inf`
    and visibly wrong, rather than silently replaced.
    """
    y = np.asarray(y, dtype=float)
    if len(y) <= lag:
        return np.nan
    diff = y[lag:] - y[:-lag]
    return float(np.sqrt(np.mean(diff**2)))


# --------------------------------------------------------------------------- #
# Holiday weeks
# --------------------------------------------------------------------------- #

# The holiday-affected window is defined once, in features.py, and shared: the
# feature masking and the normal/holiday fold split must mean the same days.
# From the event-effect table (step 3): the two days before Christmas and
# Thanksgiving run 20-70% above baseline, and the day after is still elevated.


# --------------------------------------------------------------------------- #
# The walk-forward
# --------------------------------------------------------------------------- #


def _code_digest(path: Path) -> str:
    """SHA-1 of a module's syntax tree with docstrings removed - the code, not the prose."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(
            node, ast.Module | ast.FunctionDef | ast.ClassDef | ast.AsyncFunctionDef
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
            ):
                if isinstance(body[0].value.value, str):
                    node.body = body[1:] or [ast.Pass()]
    return hashlib.sha1(ast.dump(tree).encode()).hexdigest()


# Config fields that do not change a forecast: where the files live. Leaving
# them out of the key makes a cache portable between folders and machines.
_PATH_FIELDS = ("data_dir", "cache_dir")


def _config_fields(cfg: Config) -> dict[str, str]:
    """The config as {field: repr(value)}, minus the path fields."""
    return {k: repr(v) for k, v in sorted(cfg.__dict__.items()) if k not in _PATH_FIELDS}


def _method_code(method: str) -> str:
    """
    Digest of the code a method's predictions depend on: the harness, the
    shared base (Context and helpers), the features and loader versions, and
    the module that method lives in. Editing one model's module changes only
    that method's digest, so the others' cached predictions stay valid.
    """
    from src import step4_models
    from src.models import base

    parts = [
        _code_digest(Path(__file__)),
        _code_digest(Path(base.__file__)),
        _code_digest(Path(step4_models.MODULE_OF[method].__file__)),
        str(FEATURE_VERSION),
        str(step2_data.CACHE_VERSION),
    ]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


def _method_cache_path(cfg: Config, method: str) -> Path:
    """
    One parquet per (config, method, code) under cache/predictions/, with a
    JSON sidecar recording the config fields it was built from. The sidecar
    is what lets a later config, with a new field at its default, adopt the
    run instead of repeating it.
    """
    key = repr((_config_fields(cfg), method, _method_code(method)))
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return cfg.cache_dir / "predictions" / f"{method}_{digest}.parquet"


def _adopt_cached(cfg: Config, method: str) -> Path | None:
    """
    Find an earlier run of `method` that this config can reuse.

    A run is reusable when it was built by the same code (same digest), and
    every config field the two have in common agrees, and every field the
    new config has that the old one lacks is at its dataclass default - the
    old run was made before that field existed, which is the same thing as
    the field being at its default. Adding `mask_holidays=False` to Config,
    for instance, must not throw away a twenty-minute run made without it.
    """
    from dataclasses import fields

    want = _config_fields(cfg)
    code = _method_code(method)
    defaults = {f.name: repr(f.default) for f in fields(cfg)}
    folder = cfg.cache_dir / "predictions"
    if not folder.exists():
        return None
    for meta in sorted(folder.glob(f"{method}_*.json")):
        info = json.loads(meta.read_text())
        if info.get("code") != code or info.get("method") != method:
            continue
        have = info.get("config", {})
        if any(want[k] != have[k] for k in want.keys() & have.keys()):
            continue
        if any(want[k] != defaults.get(k) for k in want.keys() - have.keys()):
            continue
        parquet = meta.with_suffix(".parquet")
        if parquet.exists():
            return parquet
    return None


def _write_cached(cfg: Config, method: str, frame: pd.DataFrame) -> Path:
    path = _method_cache_path(cfg, method)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "method": method,
                "code": _method_code(method),
                "config": _config_fields(cfg),
            },
            indent=1,
        )
    )
    return path


def _read_cached(cfg: Config, method: str) -> pd.DataFrame | None:
    path = _method_cache_path(cfg, method)
    if path.exists():
        return pd.read_parquet(path)
    adopted = _adopt_cached(cfg, method)
    if adopted is not None:
        frame = pd.read_parquet(adopted)
        _write_cached(cfg, method, frame)  # re-key under this config, once
        return frame
    return None


def run_walk_forward(
    df: pd.DataFrame,
    cfg: Config,
    methods: list[str] | None = None,
    progress: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Every forecast from every method for every series-fold, with the actuals.

    Returns one row per (series, fold, method, horizon day) with columns
    `actual`, `forecast` and `scale` (the RMSSE denominator that applies to
    that series-fold). This long table is the single source every score and
    plot is computed from.

    `methods` defaults to every benchmark and every model. Pass a subset to
    iterate on one method quickly.

    A full run takes on the order of twenty minutes on the step-7 layout
    (ARIMA is the slow one) and about seven times that on step 1, so each
    method's predictions are cached to parquet under a key that includes the
    config and the code that method depends on - see `_method_cache_path`.
    A method whose code has not changed is served from cache while the others
    are refitted. `use_cache=False` forces a full rerun.
    """
    methods = list(methods or ALL_FORECASTERS)
    unknown = set(methods) - set(ALL_FORECASTERS)
    if unknown:
        raise ValueError(
            f"unknown method(s) {sorted(unknown)}; choose from {list(ALL_FORECASTERS)}"
        )

    # One cache file per method. Only the methods without a valid file are
    # fitted, so editing XGBoost does not refit ARIMA.
    cached: dict[str, pd.DataFrame] = {}
    if use_cache:
        for name in methods:
            frame = _read_cached(cfg, name)
            if frame is not None:
                cached[name] = frame
    to_run = [m for m in methods if m not in cached]
    if not to_run:
        return _ordered(cached, methods)

    # ARIMA memoises its chosen order per series on the first fold it sees,
    # and the quantile model memoises its fit per fold. Both must belong to
    # *this* run: a second layout in the same process would otherwise inherit
    # state from a window that may reach into its own scored period.
    reset_run_state()

    last_date = df["date"].max()
    origins = cfg.fold_origins(last_date)
    holdout = cfg.holdout_start(last_date)
    matrices = supervised_matrices(df, cfg)

    rows = []
    groups = df.groupby("id", sort=True, observed=True)
    for series_id, series in tqdm(groups, desc="series", disable=not progress):
        series = series.sort_values("date").reset_index(drop=True)
        item_id, store_id = series["item_id"].iloc[0], series["store_id"].iloc[0]

        # The rows a learned model may train on: this series alone, or every
        # series that shares its pool key. Which is `Config.pool_by`'s decision.
        if cfg.pool_by is None:
            pool = matrices[series_id]
        else:
            key = series[cfg.pool_by].iloc[0]
            members = df.loc[df[cfg.pool_by] == key, "id"].unique()
            pool = pd.concat([matrices[m] for m in members], ignore_index=True)

        # "pre_holdout": one denominator for this series, from everything
        # before the first scored day. Shared by every fold and every method.
        fixed_scale = naive_scale(
            series.loc[series["date"] < holdout, "sales"], cfg.rmsse_scale_lag
        )

        for fold, origin in enumerate(origins):
            history = series[series["date"] <= origin]
            window_end = origin + pd.Timedelta(days=cfg.test_window)
            targets = series[(series["date"] > origin) & (series["date"] <= window_end)]
            if len(history) < cfg.min_train_days or len(targets) < cfg.test_window:
                continue

            if cfg.rmsse_scale_window == "per_fold":
                scale = naive_scale(history["sales"], cfg.rmsse_scale_lag)
            else:
                scale = fixed_scale

            ctx = Context(
                history=history,
                targets=targets.drop(columns=["sales"]),
                origin=origin,
                horizon=cfg.horizon,
                series_id=series_id,
                season=cfg.season,
                train_pool=pool,
                pool_by=cfg.pool_by,
                seed=cfg.seed,
            )
            actual = targets["sales"].to_numpy(dtype=float)
            dates = targets["date"].to_numpy()
            closure = targets["closure"].to_numpy(dtype=bool)
            in_window = holiday_window(targets).to_numpy(dtype=bool)
            week_kind = "holiday" if in_window.any() else "normal"

            for name in to_run:
                forecast = np.asarray(ALL_FORECASTERS[name](ctx), dtype=float)[
                    : len(targets)
                ]
                for h, (day, a, f, c, w) in enumerate(
                    zip(dates, actual, forecast, closure, in_window, strict=True), 1
                ):
                    rows.append(
                        {
                            "id": series_id,
                            "item_id": item_id,
                            "store_id": store_id,
                            "fold": fold,
                            "origin": origin,
                            "week_kind": week_kind,
                            "method": name,
                            "kind": "benchmark" if name in BENCHMARKS else "model",
                            "horizon": h,
                            "target_date": day,
                            "actual": a,
                            "forecast": f,
                            "scale": scale,
                            "closure": c,
                            "holiday_window": w,
                        }
                    )

    fresh = pd.DataFrame(rows)
    for name in to_run:
        part = fresh[fresh["method"] == name].reset_index(drop=True)
        cached[name] = part
        if use_cache:
            _write_cached(cfg, name, part)
    return _ordered(cached, methods)


def _ordered(parts: dict[str, pd.DataFrame], methods: list[str]) -> pd.DataFrame:
    """Concatenate per-method frames in the requested method order."""
    frames = [parts[m] for m in methods if m in parts and len(parts[m])]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# Scoring and tables live in src/scoring.py, re-exported here so existing
# imports keep working. They are a separate module so that editing a table
# does not change this module's code digest and invalidate the cache.
from src.scoring import (  # noqa: E402
    REFERENCE_BENCHMARK,
    improvement_over,
    rmsse_by_store,
    score_folds,
    summarise,
    win_rates,
)

if __name__ == "__main__":
    from src.step1_problem import STUDY_ITEMS
    from src.step2_data import load_panel

    cfg = Config(item_ids=STUDY_ITEMS)
    print(cfg.describe(), "\n")

    panel = load_panel(cfg, verbose=False)
    predictions = run_walk_forward(panel, cfg)
    scores = score_folds(predictions)

    print(
        f"\n{len(predictions):,} forecast-days, {len(scores):,} series-fold-method scores\n"
    )
    print("=== summary: mean RMSSE per method, best first (all / normal / holiday) ===")
    print(summarise(scores).round(3).to_string(index=False))
    for bench in (REFERENCE_BENCHMARK, "moving_average_28"):
        print(
            f"\n=== win rate and % improvement in RMSSE over {bench}, per series-fold ==="
        )
        print(improvement_over(scores, bench).round(2).to_string(index=False))
    for week in (None, "normal", "holiday"):
        label = week or "all"
        print(
            f"\n=== RMSSE by store (busiest first) x method (best first) - {label} weeks ==="
        )
        print(rmsse_by_store(scores, week).round(3).to_string())
        print(
            f"\n=== win rate: rows beat columns, share of series-folds - {label} weeks ==="
        )
        print(win_rates(scores, week).round(2).to_string())
