"""
Step 5. The harness. It walks every series through every fold, asks each
forecaster for its forecast and records what actually sold. Every table and
plot downstream comes from the one long table `run_walk_forward` returns.

Walk-forward validation (FPP §5.10). A fold is a forecast origin T plus the
`test_window` days after it. No forecaster sees anything after T. The Context
carries no future sales, and the learned models train only on rows whose
target date is at or before T.

RMSSE (FPP §5.8). The forecast's RMSE divided by the RMSE of a lag-`season`
naive forecast on the training data. 1.0 means no better than repeating the
value from a week ago, 0.8 means 20% better. The denominator belongs to the
series, which makes stores of very different size comparable. Within one
series-fold every method shares the denominator, so RMSSE never changes which
method wins a fold, only how folds and stores get aggregated. RMSE, MAE and
bias stay in the table as the plain-units view. Bias is mean(forecast - actual)
and matters on its own for replenishment.

Two kinds of week. Every score is also split into normal and holiday folds. A
fold is a holiday fold when one of its scored days falls from two days before a
major event to one day after. Both columns get reported, always. A pooled number
cannot say whether a method's advantage came from the ordinary weeks or from the
handful the calendar decides.

Closure days are not scored. Step 2 imputes them to keep the following week's
features sane, and `score_folds` drops them.
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
    # The panel version goes in the key as well. A loader change alters the
    # sales the lags and targets come from, and a matrix built on the old
    # panel must not be served for the new one.
    key = repr(
        (
            FEATURE_VERSION,
            step2_data.CACHE_VERSION,  # read through the module so a bump is always seen
            series_id,
            cfg.horizon,
            cfg.use_price,
            extent,  # first day, last day, row count. New days rebuild the matrix
        )
    )
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return cfg.cache_dir / f"supervised_v{FEATURE_VERSION}_{digest}.parquet"


def supervised_matrices(df: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    """
    One supervised matrix per series, built once and cached to parquet.

    Building takes about 8 s per series, reading back well under a second. The
    key includes FEATURE_VERSION and the series' extent, so an edited feature or
    an extra day of data rebuilds the matrix instead of serving stale rows.
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
    RMSE of a lag-`lag` naive forecast over the values given. This is the RMSSE
    denominator of FPP §5.8 in its squared form.

    Returns NaN when there is nothing to difference. A series that never changes
    gives 0. Scoring treats both cases as unscored folds.
    """
    y = np.asarray(y, dtype=float)
    if len(y) <= lag:
        return np.nan
    diff = y[lag:] - y[:-lag]
    return float(np.sqrt(np.mean(diff**2)))


# --------------------------------------------------------------------------- #
# Holiday weeks
# --------------------------------------------------------------------------- #

# The holiday window is defined once in features.py and shared with the
# normal/holiday fold split, so both mean the same days. From the event-effect
# table in step 3, the two days before Christmas and Thanksgiving run 20-70%
# above baseline and the day after is still high.


# --------------------------------------------------------------------------- #
# The walk-forward
# --------------------------------------------------------------------------- #


def _code_digest(path: Path) -> str:
    """SHA-1 of a module's syntax tree with the docstrings removed. Comments and docstrings can change without touching the cache."""
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


# Config fields that do not change a forecast, only where the files live.
# Leaving them out of the key keeps a cache portable between folders and machines.
_PATH_FIELDS = ("data_dir", "cache_dir")


def _config_fields(cfg: Config) -> dict[str, str]:
    """The config as {field: repr(value)}, minus the path fields."""
    return {k: repr(v) for k, v in sorted(cfg.__dict__.items()) if k not in _PATH_FIELDS}


def _method_code(method: str) -> str:
    """
    The digest of the code that a method's forecasts depend on. The harness,
    shared base, the method's own module, features, loader and Config all go in
    here. If one model is edited, only that model refits.
    """
    from src import features, step4_models
    from src.models import base

    # The feature and loader modules are hashed as well as their version
    # numbers. A forgotten version bump then cannot serve a stale run.
    parts = [
        _code_digest(Path(__file__)),
        _code_digest(Path(step1_problem.__file__)),  # fold layout, holdout, scale window. Known mistake: missing until item 8
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
    One parquet per (config, method, code) under cache/predictions, with a JSON
    sidecar that records the config it was built from.
    """
    key = repr((_config_fields(cfg), method, _method_code(method)))
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return cfg.cache_dir / "predictions" / f"{method}_{digest}.parquet"


def _write_cached(cfg: Config, method: str, frame: pd.DataFrame) -> Path:
    path = _method_cache_path(cfg, method)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)  # write to a temp file and rename. A kill mid-write leaves no half file
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

    One row per (series, fold, method, horizon day), with `actual`, `forecast`
    and `scale` (the RMSSE denominator for that series-fold). Every score and
    plot is computed from this table.

    `methods` defaults to every benchmark and model. Pass a subset while
    iterating on one method.

    Each method's forecasts are cached under a key made of the config and the
    code the method depends on (see `_method_cache_path`). A method whose code
    has not changed loads from the cache while the others refit.
    `use_cache=False` forces a full rerun.

    `n_jobs` is how many fits run at once, in separate processes with one
    thread each. The default is every core. It changes wall time only. The
    forecasts are the same for any value and on any machine with the same
    library versions, which is why it is not part of the cache key.
    """
    methods = list(methods or ALL_FORECASTERS)
    unknown = set(methods) - set(ALL_FORECASTERS)
    if unknown:
        raise ValueError(
            f"unknown method(s) {sorted(unknown)}; choose from {list(ALL_FORECASTERS)}"
        )

    # One cache file per method. Only methods without a valid file get fitted.
    # editing XGBoost does not refit ARIMA.
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

    # ARIMA memoises its chosen order per series and the tree models memoise
    # fits per fold or origin. All of that has to belong to this run. A second
    # layout in the same process would otherwise inherit state from a window
    # that reaches into its own scored period. Workers reset per task, the
    # parent resets here.
    reset_run_state()

    state = _run_state(df, cfg, to_run)
    n_jobs = max(1, n_jobs or os.cpu_count() or 1)

    # ARIMA chooses its order once per series on the first scored window and
    # keeps it for the year. With folds spread over processes that choice gets
    # made first and handed to every worker. Otherwise each worker would choose
    # on whatever window it saw first.
    if "arima" in to_run:
        state["arima_orders"] = _select_arima_orders(state, n_jobs, progress)

    tasks = _tasks(state)
    rows: list[dict] = []
    for chunk in _map(state, _forecast_task, tasks, n_jobs, progress, desc="folds"):
        rows.extend(chunk)

    global _STATE
    _STATE = {}  # the in-process path would leave the whole state in memory otherwise
    fresh = pd.DataFrame(rows, columns=ROW_COLUMNS)  # empty frame when every fold was skipped
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
# A task is one fold of one series when models fit per series. When they pool
# it is one fold of every series, because a pooled model is fitted once per
# origin and shared by the ten stores, and the ten have to sit in the same
# process. Each task rebuilds the Context the old serial loop built, calls every
# method in `to_run` on it and returns the rows. Tasks share nothing but the
# read-only state handed to each worker at start-up.

_STATE: dict = {}

ROW_COLUMNS = [
    "id", "item_id", "store_id", "fold", "origin", "week_kind", "method", "kind",
    "horizon", "target_date", "actual", "forecast", "scale", "closure",
    "holiday_window", "fallback",
]


def _check_actuals(df: pd.DataFrame, cached: dict[str, pd.DataFrame]) -> None:
    """
    A cached run has to match the data on disk. The key covers the config and
    the code but says nothing about the data, and a changed data file would get
    old forecasts served back with no complaint. This compares the cached
    actuals with the panel's sales and stops the run with an error if they
    differ anywhere.
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
        # The rows a learned model may train on. This series alone, or every
        # series that shares its pool key. `Config.pool_by` decides.
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
        # "pre_holdout" means one denominator per series, from everything before
        # the first scored day, shared by every fold and method.
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
    """One task per (series, fold), or one per fold when the models pool."""
    cfg = state["cfg"]
    folds = range(len(state["origins"]))
    if cfg.pool_by is None:
        return [(sid, fold) for sid in state["ids"] for fold in folds]
    return [(None, fold) for fold in folds]


def _init_worker(state: dict, pin: bool = True) -> None:
    """
    Runs once per worker. Pin every numeric library to one thread before
    XGBoost loads, because XGBoost picks its thread count at import. The
    environment variables cover libraries not loaded yet, `threadpool_limits`
    covers the ones already loaded.
    - Known mistake: pinning after the import. Twenty workers x twenty threads,
      load average 200, a 10 min gate still running after 40 min. lol
    """
    global _STATE
    _STATE = state
    if pin:  # a worker process, pinned for its whole life
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[var] = "1"
        import xgboost  # noqa: F401  - load it now, while the limit is in place
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=1)
    from src.models import arima_model

    arima_model.arima_orders.clear()
    arima_model.arima_orders.update(state["arima_orders"])


def _fold_context(series_id: str, fold: int, state: dict):
    """The Context and scoring metadata for one fold of one series. None when the fold is not scored."""
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
    """
    Call every method on one fold and return one row per forecast day. A
    method that fell back to the 28-day mean is flagged on all of its rows.
    """
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

    # Per-fold and per-origin memos belong to this task only. A fresh task must
    # never reuse a fit and a long-lived worker must not hoard them.
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
    """Choose the ARIMA order for one series on its first scored window, as the old serial loop did."""
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
    """Choose the ARIMA order for every series before the folds run. Fails if a series with scored folds got none."""
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
    """Apply `fn` to `items` in order, in `n_jobs` worker processes, or in-process when n_jobs is 1."""
    n_jobs = min(n_jobs, max(1, len(items)))
    if n_jobs == 1:
        from threadpoolctl import threadpool_limits

        _init_worker(state, pin=False)
        with threadpool_limits(limits=1):  # only for the duration of the run
            for item in tqdm(items, desc=desc, disable=not progress):
                yield fn(item)
        return
    import multiprocessing as mp

    from threadpoolctl import threadpool_limits

    # fork rather than the Python 3.14 default, forkserver. Workers inherit the
    # state with no pickling and a calling script needs no __main__ guard.
    # Safe here because the parent does no threaded numerics of its own. Its
    # libraries are pinned to one thread before the fork and tqdm's monitor
    # thread is switched off.
    tqdm.monitor_interval = 0
    threadpool_limits(limits=1)
    # Python warns that forking a threaded process can deadlock. The only
    # threads here are idle library pools pinned to one.
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
    A block for the top of a findings file. It ties every number there to the
    branch, commit, config and cache files that made it, so anyone can check
    out the commit, rebuild the config and compare.

    Call it in the same checkout at the time of the run. Called later, the
    branch and commit lines describe where the block was written and have to
    be corrected by hand from the run's history.
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


# Scoring and tables live in src/scoring.py and are re-exported here so the
# old imports keep working. They are a separate module on purpose. Editing a
# table must not change this module's digest and invalidate the cache.
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
