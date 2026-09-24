"""
The run registry: every run on record, so a change is measured against
the last one instead of assumed.

The cache already keeps every forecast, keyed on configuration, method and
code. What it lacks is a list: which runs exist, when and at what commit
they were made, and how they scored. The registry is that list, in a
SQLite file beside the cache (`cache/registry.sqlite`), with the same two
tables the planned SQL Server results store will have:

  run         one row per cached run (run_id = the cache file's stem):
              method, layout, item, pooling, code digest, git state, time,
              headline scores against the seasonal naive and the 28-day mean
  fold_score  one row per store-origin of that run: RMSSE, bias, MAE, RMSE,
              the store's level that week, the kind of week

Everything in it is derived from the cache and the git history; it is never
edited by hand. Rebuild it at any time with `backfill()`.

Recording happens through `record()`, which wraps the harness rather than
living inside it: the harness file is part of every cache key, so a change
there would invalidate every cached run. Use `python -m src.run` for new
runs; `python -m src.registry backfill` indexes the runs already on disk.

Comparisons are paired: the same store and the same origin under two runs,
which is the only comparison that means anything when folds differ in
difficulty by a factor of three.
"""

from __future__ import annotations

import ast
import json
import sqlite3
import subprocess
import sys
import time
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.scoring import REFERENCE_BENCHMARK, improvement_over, score_folds, summarise
from src.step1_problem import PROJECT_ROOT, Config
from src.step5_evaluate import (
    _config_fields,
    _method_cache_path,
    _method_code,
    _read_cached,
    run_walk_forward,
)
from src.suites import suite_of

STRESS_BENCHMARK = "moving_average_28"

SCHEMA = """
CREATE TABLE IF NOT EXISTS run (
    run_id        TEXT PRIMARY KEY,   -- cache file stem: <method>_<digest>
    method        TEXT NOT NULL,
    kind          TEXT NOT NULL,      -- benchmark | model
    suite         TEXT,               -- dev | weekly | everyday | NULL
    item_ids      TEXT NOT NULL,
    store_ids     TEXT NOT NULL,      -- '' = all ten
    pool_by       TEXT,
    fold_step     INTEGER NOT NULL,
    n_folds       INTEGER NOT NULL,
    horizon       INTEGER NOT NULL,
    seed          INTEGER NOT NULL,
    config_json   TEXT NOT NULL,      -- every Config field except paths
    code_digest   TEXT NOT NULL,
    code_current  INTEGER NOT NULL,   -- 1 if the digest matches the code now
    git_branch    TEXT,
    git_commit    TEXT,
    git_dirty     INTEGER,
    recorded_at   TEXT NOT NULL,      -- ISO 8601, UTC
    source        TEXT NOT NULL,      -- run | backfill | rekey
    run_seconds   REAL,
    note          TEXT,
    n_scored      INTEGER, n_unscored INTEGER, n_fallback INTEGER,
    rmsse_mean    REAL, rmsse_median REAL, rmsse_normal REAL, rmsse_holiday REAL,
    bias_mean     REAL,
    win_vs_ref    REAL, impr_mean REAL, impr_q1 REAL, impr_median REAL, impr_q3 REAL,
    win_vs_stress REAL, impr_median_stress REAL,
    cache_file    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fold_score (
    run_id      TEXT NOT NULL REFERENCES run(run_id),
    id          TEXT NOT NULL,
    item_id     TEXT NOT NULL,
    store_id    TEXT NOT NULL,
    fold        INTEGER NOT NULL,
    origin      TEXT NOT NULL,
    week_kind   TEXT NOT NULL,
    rmsse       REAL, bias REAL, mae REAL, rmse REAL,
    mean_actual REAL,
    n_days      INTEGER,
    unscored    INTEGER,
    fallback    INTEGER,
    PRIMARY KEY (run_id, id, fold)
);
CREATE INDEX IF NOT EXISTS fold_score_origin ON fold_score (id, origin);
"""


def registry_path(cfg: Config | None = None) -> Path:
    return (cfg.cache_dir if cfg else Path("cache")) / "registry.sqlite"


def connect(cfg: Config | None = None) -> sqlite3.Connection:
    path = registry_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    # columns added after the first release; harmless when already present
    for table, column, kind in (("run", "n_unscored", "INTEGER"), ("run", "n_fallback", "INTEGER"),
                                ("fold_score", "unscored", "INTEGER"), ("fold_score", "fallback", "INTEGER")):
        try:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
        except sqlite3.OperationalError:
            pass
    return con


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, cwd=PROJECT_ROOT).strip()
    except Exception:
        return None


def _git_state() -> dict:
    return {
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": int(bool(_git("status", "--porcelain"))) if _git("status", "--porcelain") is not None else None,
    }


def _benchmark_scores(cfg: Config, panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    The reference and stress benchmarks scored under this config's layout.
    Benchmarks never pool, so a pooled config takes them from the per-series
    config with the same layout. Cheap to fit if not cached (seconds), from
    `panel` when given, else from the loader - but never for a config with
    no item chosen, which would mean loading the whole dataset.
    """
    per_series = Config(**{**cfg.__dict__, "pool_by": None})
    frames = []
    for method in (REFERENCE_BENCHMARK, STRESS_BENCHMARK):
        cached = _read_cached(per_series, method)
        if cached is None:
            if panel is None:
                if not per_series.item_ids:
                    continue
                from src.step2_data import load_panel

                panel = load_panel(per_series, verbose=False)
            cached = run_walk_forward(panel, per_series, methods=[method], progress=False)
        frames.append(cached)
    if not frames:
        return pd.DataFrame(columns=["method"])
    return score_folds(pd.concat(frames, ignore_index=True))


def _headline(scores: pd.DataFrame, method: str, bench: pd.DataFrame) -> dict:
    """The run's summary row, with paired improvement over both benchmarks."""
    row = summarise(scores).set_index("method").loc[method]
    out = {
        "n_scored": int(row["n_folds"]),
        "n_unscored": int(row["n_unscored"]),
        "n_fallback": int(row["n_fallback"]),
        "rmsse_mean": float(row["rmsse_mean"]),
        "rmsse_median": float(row["rmsse_median"]),
        "rmsse_normal": float(row["rmsse_normal"]),
        "rmsse_holiday": float(row["rmsse_holiday"]),
        "bias_mean": float(row["bias_mean"]),
    }
    both = pd.concat([scores, bench[bench["method"] != method]], ignore_index=True)
    for benchmark, prefix in ((REFERENCE_BENCHMARK, "ref"), (STRESS_BENCHMARK, "stress")):
        if benchmark == method or benchmark not in set(both["method"]):
            continue
        imp = improvement_over(both, benchmark).set_index("method")
        if method not in imp.index:
            continue
        r = imp.loc[method]
        out[f"win_vs_{prefix}"] = float(r["win_rate"])
        out[f"impr_median_{prefix}" if prefix == "stress" else "impr_median"] = float(r["median_pct"])
        if prefix == "ref":
            out["impr_mean"] = float(r["mean_improvement_pct"])
            out["impr_q1"] = float(r["q1_pct"])
            out["impr_q3"] = float(r["q3_pct"])
    return out


def _insert(con: sqlite3.Connection, run_row: dict, scores: pd.DataFrame) -> bool:
    """
    A run is recorded once. Recording the same run again (same forecasts,
    same code) keeps the first row - its time and commit describe when the
    forecasts were made - and only appends the new note. Returns whether a
    row was written.
    """
    existing = con.execute("SELECT note FROM run WHERE run_id = ?", (run_row["run_id"],)).fetchone()
    if existing is not None:
        if run_row.get("note"):
            merged = f"{existing[0]} | {run_row['note']}" if existing[0] else run_row["note"]
            con.execute("UPDATE run SET note = ? WHERE run_id = ?", (merged, run_row["run_id"]))
            con.commit()
        return False
    cols = ", ".join(run_row)
    marks = ", ".join("?" for _ in run_row)
    con.execute(f"INSERT INTO run ({cols}) VALUES ({marks})", list(run_row.values()))
    rows = scores[["id", "item_id", "store_id", "fold", "origin", "week_kind",
                   "rmsse", "bias", "mae", "rmse", "mean_actual", "n_days", "unscored", "fallback"]].copy()
    rows.insert(0, "run_id", run_row["run_id"])
    rows["origin"] = pd.to_datetime(rows["origin"]).dt.strftime("%Y-%m-%d")
    rows["rmsse"] = rows["rmsse"].replace([np.inf, -np.inf], np.nan)
    rows["unscored"] = rows["unscored"].astype(int)
    rows["fallback"] = rows["fallback"].astype(int)
    con.executemany(
        "INSERT INTO fold_score VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows.itertuples(index=False, name=None),
    )
    con.commit()
    return True


def _run_row(cfg: Config, method: str, predictions: pd.DataFrame, *, source: str,
             recorded_at: str, run_seconds: float | None, note: str | None,
             code_digest: str | None = None, git: dict | None = None,
             panel: pd.DataFrame | None = None) -> tuple[dict, pd.DataFrame]:
    path = _method_cache_path(cfg, method)
    digest_now = _method_code(method)
    digest = code_digest or digest_now
    scores = score_folds(predictions)
    bench = _benchmark_scores(cfg, panel)
    row = {
        "run_id": path.stem if digest == digest_now else f"{method}_{digest[:12]}",
        "method": method,
        "kind": str(predictions["kind"].iloc[0]),
        "suite": suite_of(cfg),
        "item_ids": ",".join(cfg.item_ids),
        "store_ids": ",".join(cfg.store_ids),
        "pool_by": cfg.pool_by,
        "fold_step": cfg.fold_step,
        "n_folds": cfg.n_folds,
        "horizon": cfg.horizon,
        "seed": cfg.seed,
        "config_json": json.dumps(_config_fields(cfg), sort_keys=True),
        "code_digest": digest,
        "code_current": int(digest == digest_now),
        **(git or {"git_branch": None, "git_commit": None, "git_dirty": None}),
        "recorded_at": recorded_at,
        "source": source,
        "run_seconds": run_seconds,
        "note": note,
        **_headline(scores, method, bench),
        "cache_file": path.name,
    }
    return row, scores


def record(cfg: Config, methods: list[str], note: str | None = None,
           progress: bool = True) -> list[str]:
    """
    Run the harness for `methods` under `cfg` (cached methods load instantly)
    and register each. Returns the run ids. This is the entry point for new
    runs; see `python -m src.run`.
    """
    from src.step2_data import load_panel

    df = load_panel(cfg, verbose=False)
    con = connect(cfg)
    ids = []
    for method in methods:
        t = time.time()
        predictions = run_walk_forward(df, cfg, methods=[method], progress=progress)
        seconds = time.time() - t
        row, scores = _run_row(
            cfg, method, predictions, source="run",
            recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            run_seconds=round(seconds, 1), note=note, git=_git_state(), panel=df,
        )
        _insert(con, row, scores)
        ids.append(row["run_id"])
    con.close()
    return ids


# --------------------------------------------------------------------------- #
# Back-fill from the cache on disk
# --------------------------------------------------------------------------- #


def _config_from_sidecar(info: dict, base: Config) -> Config:
    """
    Rebuild a Config from a sidecar's {field: repr(value)}. Missing fields
    take defaults; the path fields, which sidecars leave out, come from `base`.
    """
    known = {f.name for f in fields(Config)}
    kwargs = {k: ast.literal_eval(v) for k, v in info["config"].items() if k in known}
    return Config(**kwargs, data_dir=base.data_dir, cache_dir=base.cache_dir)


def refresh_code_current(cfg: Config | None = None) -> int:
    """
    Recompute `code_current` for every row: 1 where the run's code digest is
    what the code computes now for that method. Called by `backfill`, and
    worth calling after any code change, since the column is a snapshot.
    """
    con = connect(cfg)
    rows = con.execute("SELECT run_id, method, code_digest FROM run").fetchall()
    now = {}
    changed = 0
    for run_id, method, digest in rows:
        if method not in now:
            try:
                now[method] = _method_code(method)
            except KeyError:
                now[method] = None
        current = int(digest == now[method])
        cur = con.execute("UPDATE run SET code_current = ? WHERE run_id = ? AND code_current != ?",
                          (current, run_id, current))
        changed += cur.rowcount
    con.commit()
    con.close()
    return changed


def backfill(cfg: Config | None = None, verbose: bool = True) -> int:
    """
    Index every cached run under cache/predictions. Runs made under older
    code are kept with code_current = 0. Idempotent: rows are replaced.
    """
    cfg = cfg or Config()
    refresh_code_current(cfg)
    folder = cfg.cache_dir / "predictions"
    con = connect(cfg)
    have = {r[0] for r in con.execute("SELECT run_id FROM run")}
    n = 0
    for meta in sorted(folder.glob("*.json")):
        if meta.stem in have:
            continue
        info = json.loads(meta.read_text())
        run_cfg = _config_from_sidecar(info, cfg)
        method = info["method"]
        parquet = meta.with_suffix(".parquet")
        if not parquet.exists():
            continue
        predictions = pd.read_parquet(parquet)
        recorded = datetime.fromtimestamp(parquet.stat().st_mtime, tz=timezone.utc)
        row, scores = _run_row(
            run_cfg, method, predictions, source="backfill",
            recorded_at=recorded.isoformat(timespec="seconds"), run_seconds=None,
            note=None, code_digest=info["code"],
        )
        row["run_id"] = meta.stem  # the file on disk names the run
        row["cache_file"] = parquet.name
        if not _insert(con, row, scores):
            continue
        n += 1
        if verbose:
            print(f"  {row['run_id']:<40} {row['suite'] or '-':<9} {row['item_ids']:<12} "
                  f"pool={row['pool_by'] or '-':<8} rmsse={row['rmsse_mean']:.3f}")
    con.close()
    return n


# --------------------------------------------------------------------------- #
# Reading and comparing
# --------------------------------------------------------------------------- #


def runs(cfg: Config | None = None, where: str = "1=1", params: tuple = ()) -> pd.DataFrame:
    con = connect(cfg)
    try:
        return pd.read_sql_query(
            f"SELECT * FROM run WHERE {where} ORDER BY recorded_at", con, params=params
        )
    finally:
        con.close()


def predecessor(run_id: str, cfg: Config | None = None) -> str | None:
    """The most recent earlier run of the same method on the same suite, item, stores and pooling."""
    con = connect(cfg)
    try:
        me = con.execute("SELECT * FROM run WHERE run_id = ?", (run_id,)).fetchone()
        if me is None:
            raise KeyError(run_id)
        cols = [d[0] for d in con.execute("SELECT * FROM run LIMIT 0").description]
        me = dict(zip(cols, me))
        row = con.execute(
            """SELECT run_id FROM run
               WHERE method = ? AND config_json = ? AND run_id != ? AND recorded_at < ?
               ORDER BY recorded_at DESC LIMIT 1""",
            (me["method"], me["config_json"], run_id, me["recorded_at"]),
        ).fetchone()
        return row[0] if row else None
    finally:
        con.close()


def compare(run_a: str, run_b: str, cfg: Config | None = None) -> dict:
    """
    Paired comparison of run B against run A on the store-origins they share:
    win rate (B's RMSSE lower), and mean, quartiles and median of the
    percentage reduction in RMSSE. A difference inside the fold-to-fold noise
    shows up as a win rate near 0.5 and a median near zero.
    """
    con = connect(cfg)
    try:
        q = "SELECT id, origin, rmsse, bias FROM fold_score WHERE run_id = ?"
        a = pd.read_sql_query(q, con, params=(run_a,))
        b = pd.read_sql_query(q, con, params=(run_b,))
    finally:
        con.close()
    pair = a.merge(b, on=["id", "origin"], suffixes=("_a", "_b")).dropna(subset=["rmsse_a", "rmsse_b"])
    if pair.empty:
        raise ValueError("the two runs share no store-origins")
    undefined = pair["rmsse_a"] <= 0  # a percentage change from zero is undefined
    gain = ((pair["rmsse_a"] - pair["rmsse_b"]) / pair["rmsse_a"] * 100)[~undefined]
    return {
        "a": run_a,
        "b": run_b,
        "n_paired": int(len(pair)),
        "n_undefined": int(undefined.sum()),
        "win_rate_b": float((pair["rmsse_b"] < pair["rmsse_a"]).mean()),
        "rmsse_a": float(pair["rmsse_a"].mean()),
        "rmsse_b": float(pair["rmsse_b"].mean()),
        "bias_a": float(pair["bias_a"].mean()),
        "bias_b": float(pair["bias_b"].mean()),
        "impr_mean_pct": float(gain.mean()),
        "impr_q1_pct": float(gain.quantile(0.25)),
        "impr_median_pct": float(gain.median()),
        "impr_q3_pct": float(gain.quantile(0.75)),
    }


def format_comparison(c: dict) -> str:
    return (
        f"{c['b']}  vs  {c['a']}   ({c['n_paired']} paired store-origins)\n"
        f"  RMSSE  {c['rmsse_a']:.3f} -> {c['rmsse_b']:.3f}     bias  {c['bias_a']:+.2f} -> {c['bias_b']:+.2f}\n"
        f"  B better in {c['win_rate_b']:.0%} of store-origins; "
        f"improvement mean {c['impr_mean_pct']:.1f}%, quartiles {c['impr_q1_pct']:.1f}% / "
        f"{c['impr_median_pct']:.1f}% / {c['impr_q3_pct']:.1f}%"
    )


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #

_USAGE = """usage: python -m src.registry <command> [args]

  backfill                 index every cached run on disk (idempotent)
  refresh                  recompute code_current after a code change
  list [suite]             runs on record, newest last (optionally one suite)
  show <run_id>            one run's row
  compare <run_a> <run_b>  paired comparison of B against A
  last <run_id>            compare a run against its predecessor
"""


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(_USAGE)
        return 0
    cmd, args = argv[0], argv[1:]
    pd.set_option("display.width", 200)
    if cmd == "backfill":
        n = backfill()
        print(f"indexed {n} new runs")
    elif cmd == "refresh":
        print(f"code_current changed on {refresh_code_current()} rows")
    elif cmd == "list":
        where, params = ("suite = ?", (args[0],)) if args else ("1=1", ())
        cols = ["run_id", "suite", "item_ids", "pool_by", "code_current", "git_commit",
                "recorded_at", "rmsse_mean", "bias_mean", "win_vs_ref", "impr_median", "note"]
        print(runs(where=where, params=params)[cols].to_string(index=False))
    elif cmd == "show":
        print(runs(where="run_id = ?", params=(args[0],)).T.to_string(header=False))
    elif cmd == "compare":
        print(format_comparison(compare(args[0], args[1])))
    elif cmd == "last":
        prev = predecessor(args[0])
        if prev is None:
            print(f"{args[0]}: no earlier run of the same method on the same layout")
            return 1
        print(format_comparison(compare(prev, args[0])))
    else:
        print(_USAGE)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
