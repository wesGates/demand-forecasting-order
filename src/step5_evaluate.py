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
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src import step2_data
from src.features import FEATURE_VERSION, build_supervised, holiday_window
from src.step1_problem import PROJECT_ROOT, Config
from src import step1_problem
from src.models import base
from src.step4_models import ALL_FORECASTERS, BENCHMARKS, Context, reset_run_state

# --------------------------------------------------------------------------- #
# Supervised matrices, cached
# --------------------------------------------------------------------------- #


def _supervised_path(cfg: Config, series_id: str, extent: tuple = ()):
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
            extent,  # first day, last day, rows: new days must rebuild the matrix
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
        extent = (str(series["date"].min().date()), str(series["date"].max().date()), len(series))
        path = _supervised_path(cfg, series_id, extent)
        if path.exists():
            out[series_id] = pd.read_parquet(path)
        else:
            matrix = build_supervised(
                series,
                cfg.horizon,
                use_price=cfg.use_price,
            )
            tmp = path.with_suffix(".tmp")
            matrix.to_parquet(tmp, index=False)
            tmp.replace(path)
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
    from src import features, step4_models
    from src.models import base

    # The feature and loader modules are hashed too, not only their manual
    # version numbers: an edit to either changes what a model trains on, and
    # a forgotten version bump must not serve a stale run.
    parts = [
        _code_digest(Path(__file__)),
        _code_digest(Path(step1_problem.__file__)),  # fold layout, holdout, scale window
        _code_digest(Path(base.__file__)),
        _code_digest(Path(step4_models.MODULE_OF[method].__file__)),
        _code_digest(Path(features.__file__)),
        _code_digest(Path(step2_data.__file__)),
        str(FEATURE_VERSION),
        str(step2_data.CACHE_VERSION),
    ]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


def _method_cache_path(cfg: Config, method: str) -> Path:
    """
    One parquet per (config, method, code) under cache/predictions/, with a
    JSON sidecar recording the config fields it was built from.
    """
    key = repr((_config_fields(cfg), method, _method_code(method)))
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return cfg.cache_dir / "predictions" / f"{method}_{digest}.parquet"


def _write_cached(cfg: Config, method: str, frame: pd.DataFrame) -> Path:
    path = _method_cache_path(cfg, method)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)  # atomic: a kill mid-write leaves no half file behind
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
    if path.exists() and path.with_suffix(".json").exists():
        return pd.read_parquet(path)
    return None


def run_walk_forward(
    df: pd.DataFrame,
    cfg: Config,
    methods: list[str] | None = None,
    progress: bool = True,
    use_cache: bool = True,
    n_jobs: int | None = None,
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

    `n_jobs` is how many fits run at once, in separate processes, one thread
    each (default: every core). It changes wall time only: every fit is
    single-threaded whatever `n_jobs` is, so the forecasts are the same for
    any value, and the same on any machine with the same library versions.
    Parallelism is therefore not part of the cache key.
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
    if cached:
        _check_actuals(df, cached)
    to_run = [m for m in methods if m not in cached]
    if not to_run:
        return _ordered(cached, methods)

    # ARIMA memoises its chosen order per series on the first fold it sees,
    # and the tree models memoise fits per fold or per origin. All of it must
    # belong to *this* run: a second layout in the same process would
    # otherwise inherit state from a window that may reach into its own
    # scored period. Workers reset per task; the parent resets here.
    reset_run_state()

    state = _run_state(df, cfg, to_run)
    n_jobs = max(1, n_jobs or os.cpu_count() or 1)

    # ARIMA's order is chosen once per series, on that series' first scored
    # window, and held for the year. With folds spread over processes that
    # choice has to be made first and handed to every worker, or each would
    # choose its own on whatever window it saw first.
    if "arima" in to_run:
        state["arima_orders"] = _select_arima_orders(state, n_jobs, progress)

    tasks = _tasks(state)
    rows: list[dict] = []
    for chunk in _map(state, _forecast_task, tasks, n_jobs, progress, desc="folds"):
        rows.extend(chunk)

    global _STATE
    _STATE = {}  # the in-process path leaves the state behind otherwise
    fresh = pd.DataFrame(rows, columns=ROW_COLUMNS)  # empty when every fold was skipped
    for name in to_run:
        part = (
            fresh[fresh["method"] == name]
            .sort_values(["id", "fold", "horizon"], kind="stable")
            .reset_index(drop=True)
        )
        cached[name] = part
        if use_cache:
            _write_cached(cfg, name, part)
    return _ordered(cached, methods)


# --------------------------------------------------------------------------- #
# The work, split into tasks that run in parallel
# --------------------------------------------------------------------------- #
#
# A task is one fold of one series when models fit per series, or one fold of
# every series when they pool: a pooled model is fitted once per origin and
# shared by the ten stores, so the ten have to sit in the same process. Each
# task rebuilds exactly the Context the serial loop built, calls every method
# in `to_run` on it, and returns the rows. Nothing is shared between tasks
# except the read-only state handed to each worker at start-up.

_STATE: dict = {}

ROW_COLUMNS = [
    "id", "item_id", "store_id", "fold", "origin", "week_kind", "method", "kind",
    "horizon", "target_date", "actual", "forecast", "scale", "closure",
    "holiday_window", "fallback",
]


def _check_actuals(df: pd.DataFrame, cached: dict[str, pd.DataFrame]) -> None:
    """
    A cached run must describe the data on disk now. The key names the
    configuration and the code, not the data, so a changed data file (or a
    cache copied next to different data) would otherwise be served without
    a word. The actuals in the cache and the panel's sales must agree.
    """
    sales = df.set_index(["id", "date"])["sales"]
    for name, frame in cached.items():
        if len(frame) == 0:
            continue
        idx = pd.MultiIndex.from_arrays([frame["id"].astype(str), pd.to_datetime(frame["target_date"])])
        expected = sales.reindex(idx).to_numpy(dtype=float)
        got = frame["actual"].to_numpy(dtype=float)
        if np.isnan(expected).any() or not np.allclose(expected, got, equal_nan=True):
            raise ValueError(
                f"the cached {name!r} run does not match the data on disk: its actuals differ "
                "from the panel's sales. The data changed, or this cache belongs to other data. "
                "Delete the cached file or rebuild with use_cache=False."
            )


def _run_state(df: pd.DataFrame, cfg: Config, to_run: list[str]) -> dict:
    """Everything a worker needs, built once: series, training pools, scales, origins."""
    last_date = df["date"].max()
    origins = list(cfg.fold_origins(last_date))
    holdout = cfg.holdout_start(last_date)
    matrices = supervised_matrices(df, cfg)

    series: dict[str, pd.DataFrame] = {}
    pools: dict[str, pd.DataFrame] = {}
    scales: dict[str, float] = {}
    shared_pools: dict = {}
    for series_id, frame in df.groupby("id", sort=True, observed=True):
        frame = frame.sort_values("date").reset_index(drop=True)
        series[series_id] = frame
        # The rows a learned model may train on: this series alone, or every
        # series that shares its pool key. Which is `Config.pool_by`'s decision.
        if cfg.pool_by is None:
            pools[series_id] = matrices[series_id]
        else:
            key = frame[cfg.pool_by].iloc[0]
            if key not in shared_pools:
                members = df.loc[df[cfg.pool_by] == key, "id"].unique()
                shared_pools[key] = pd.concat(
                    [matrices[m] for m in members], ignore_index=True
                )
            pools[series_id] = shared_pools[key]
        # "pre_holdout": one denominator for this series, from everything
        # before the first scored day. Shared by every fold and every method.
        scales[series_id] = naive_scale(
            frame.loc[frame["date"] < holdout, "sales"], cfg.rmsse_scale_lag
        )
    return {
        "cfg": cfg,
        "to_run": list(to_run),
        "ids": list(series),
        "series": series,
        "pools": pools,
        "scales": scales,
        "origins": origins,
        "holdout": holdout,
        "arima_orders": {},
    }


def _tasks(state: dict) -> list[tuple]:
    cfg = state["cfg"]
    folds = range(len(state["origins"]))
    if cfg.pool_by is None:
        return [(sid, fold) for sid in state["ids"] for fold in folds]
    return [(None, fold) for fold in folds]


def _init_worker(state: dict, pin: bool = True) -> None:
    """
    Runs once per worker process: take the state and pin every numerical
    library to one thread. The environment variables catch libraries not yet
    loaded (XGBoost imports lazily, and its OpenMP pool reads them at load);
    `threadpool_limits` catches the ones already loaded. Both are needed:
    with twenty workers each spawning twenty OpenMP threads, the machine ran
    four hundred busy-waiting threads and crawled.
    """
    global _STATE
    _STATE = state
    if pin:  # a worker process: pin for its whole life
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[var] = "1"
        import xgboost  # noqa: F401  - load it now, under the limit
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=1)
    from src.models import arima_model

    arima_model.arima_orders.clear()
    arima_model.arima_orders.update(state["arima_orders"])


def _fold_context(series_id: str, fold: int, state: dict):
    """The Context and scoring metadata for one fold of one series, or None if it is not scored."""
    cfg = state["cfg"]
    series = state["series"][series_id]
    origin = state["origins"][fold]
    history = series[series["date"] <= origin]
    window_end = origin + pd.Timedelta(days=cfg.test_window)
    targets = series[(series["date"] > origin) & (series["date"] <= window_end)]
    if len(history) < cfg.min_train_days or len(targets) < cfg.test_window:
        return None
    if cfg.rmsse_scale_window == "per_fold":
        scale = naive_scale(history["sales"], cfg.rmsse_scale_lag)
    else:
        scale = state["scales"][series_id]
    ctx = Context(
        history=history,
        targets=targets.drop(columns=["sales"]),
        origin=origin,
        horizon=cfg.horizon,
        series_id=series_id,
        season=cfg.season,
        train_pool=state["pools"][series_id],
        pool_by=cfg.pool_by,
        seed=cfg.seed,
    )
    in_window = holiday_window(targets).to_numpy(dtype=bool)
    meta = {
        "id": series_id,
        "item_id": series["item_id"].iloc[0],
        "store_id": series["store_id"].iloc[0],
        "fold": fold,
        "origin": origin,
        "week_kind": "holiday" if in_window.any() else "normal",
        "scale": scale,
        "actual": targets["sales"].to_numpy(dtype=float),
        "dates": targets["date"].to_numpy(),
        "closure": targets["closure"].to_numpy(dtype=bool),
        "in_window": in_window,
    }
    return ctx, meta


def _forecast_rows(ctx: Context, meta: dict, to_run: list[str]) -> list[dict]:
    rows = []
    for name in to_run:
        base.fallbacks.clear()
        forecast = np.asarray(ALL_FORECASTERS[name](ctx), dtype=float)[: len(meta["actual"])]
        fallback = bool(base.fallbacks)
        for h, (day, a, f, c, w) in enumerate(
            zip(meta["dates"], meta["actual"], forecast, meta["closure"], meta["in_window"], strict=True), 1
        ):
            rows.append(
                {
                    "id": meta["id"],
                    "item_id": meta["item_id"],
                    "store_id": meta["store_id"],
                    "fold": meta["fold"],
                    "origin": meta["origin"],
                    "week_kind": meta["week_kind"],
                    "method": name,
                    "kind": "benchmark" if name in BENCHMARKS else "model",
                    "horizon": h,
                    "target_date": day,
                    "actual": a,
                    "forecast": f,
                    "scale": meta["scale"],
                    "closure": c,
                    "holiday_window": w,
                    "fallback": fallback,
                }
            )
    return rows


def _forecast_task(task: tuple) -> list[dict]:
    """One task: a fold of one series, or (pooled) a fold of every series."""
    from src.models import xgboost_model, xgboost_quantile, xgboost_relative

    # Per-fold and per-origin memos belong to this task only: a fresh task
    # must never reuse a fit, and a long-lived worker must not hoard them.
    xgboost_model.reset()
    xgboost_quantile.reset()
    xgboost_relative.reset()

    state = _STATE
    series_id, fold = task
    ids = state["ids"] if series_id is None else [series_id]
    rows: list[dict] = []
    for sid in ids:
        built = _fold_context(sid, fold, state)
        if built is None:
            continue
        ctx, meta = built
        rows.extend(_forecast_rows(ctx, meta, state["to_run"]))
    return rows


def _first_arima_order(series_id: str):
    """Choose ARIMA's order for one series on its first scored window, as the serial loop did."""
    from src.models import arima_model

    state = _STATE
    for fold in range(len(state["origins"])):
        built = _fold_context(series_id, fold, state)
        if built is None:
            continue
        ctx, _ = built
        arima_model.arima_orders.pop(series_id, None)
        arima_model.fit_predict_arima(ctx)
        return series_id, arima_model.arima_orders.get(series_id), True
    return series_id, None, False


def _select_arima_orders(state: dict, n_jobs: int, progress: bool) -> dict:
    orders = {}
    for sid, order, scored in _map(state, _first_arima_order, state["ids"], n_jobs, progress, desc="arima orders"):
        if scored and order is None:
            raise RuntimeError(
                f"ARIMA could not choose an order for {sid} on its first scored window; "
                "refusing to let each worker choose its own on a later window"
            )
        if order is not None:
            orders[sid] = order
    return orders


def _map(state: dict, fn, items: list, n_jobs: int, progress: bool, desc: str):
    """Apply `fn` to `items` in order, in `n_jobs` worker processes (or in-process when 1)."""
    n_jobs = min(n_jobs, max(1, len(items)))
    if n_jobs == 1:
        from threadpoolctl import threadpool_limits

        _init_worker(state, pin=False)
        with threadpool_limits(limits=1):  # for the duration of the run only
            for item in tqdm(items, desc=desc, disable=not progress):
                yield fn(item)
        return
    import multiprocessing as mp

    from threadpoolctl import threadpool_limits

    # fork, not the 3.14 default forkserver: workers inherit the state
    # without pickling 50 MB per worker, and a calling script needs no
    # `__main__` guard. Fork is safe here because the parent runs no
    # threaded numerics of its own: its libraries are pinned to one
    # thread before the fork, and tqdm's monitor thread is switched off.
    tqdm.monitor_interval = 0
    threadpool_limits(limits=1)
    # Python warns that forking a process with threads can deadlock; the
    # only threads here are idle library pools, pinned to one above.
    warnings.filterwarnings("ignore", message=".*multi-threaded.*fork.*")
    ctx = mp.get_context("fork")
    chunksize = max(1, len(items) // (n_jobs * 8))
    with ctx.Pool(n_jobs, initializer=_init_worker, initargs=(state,)) as pool:
        for result in tqdm(
            pool.imap(fn, items, chunksize=chunksize), total=len(items), desc=desc, disable=not progress
        ):
            yield result


def provenance(cfg: Config, methods: list[str] | None = None) -> str:
    """
    A block for the top of a findings file that ties every number in it to
    the branch, commit, config and cache files that produced it. Anyone can
    check out the commit, rebuild the config from the line printed here, and
    either read the same cache file or recompute and compare.

    Call it in the same checkout and at the same time as the run, or the
    branch and commit lines describe where the block was written rather
    than where the predictions were made. Stamping a file later means
    writing those two lines by hand from the run's history.
    """
    import subprocess

    def git(*args):
        try:
            return subprocess.check_output(
                ["git", *args], text=True, cwd=PROJECT_ROOT
            ).strip()
        except Exception:  # not a git checkout
            return "?"

    methods = list(methods or ALL_FORECASTERS)
    fields = ", ".join(f"{k}={v}" for k, v in _config_fields(cfg).items())
    lines = [
        f"branch: {git('rev-parse', '--abbrev-ref', 'HEAD')}",
        f"commit: {git('rev-parse', '--short', 'HEAD')}"
        + (
            " (working tree has uncommitted changes)"
            if git("status", "--porcelain")
            else ""
        ),
        f"config: Config({fields})",
        "cache files:",
    ]
    for m in methods:
        path = _method_cache_path(cfg, m)
        lines.append(f"  {m}: {path.name}" + ("" if path.exists() else "  (not cached)"))
    return "\n".join(lines)


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
