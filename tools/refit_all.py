"""Refit every reported run and register each one under a note.
Used after a change that touches forecasts (item 7, item 8); the note tells
the registry which refit a run belongs to. Stages run one after another (each already uses every
core); a stage that takes more than twice its estimate is killed and logged as
TIMEOUT so the cause can be found before anything else runs. Old cache files
are never touched: the new harness writes under new keys.
Run from the repo root: PYTHONPATH=. python tools/refit_all.py "<note>" > refit.log"""
import subprocess, sys, time
from datetime import datetime

POINT = ["mean", "naive", "seasonal_naive", "seasonal_naive_364", "drift", "moving_average_28",
         "ets", "arima", "xgboost", "xgboost_rel"]
POOLED = ["xgboost", "xgboost_rel"]
import sys
NOTE = sys.argv[1] if len(sys.argv) > 1 else "item 7 refit: parallel harness, one thread per fit"

# (suite, item, pool, methods, estimate in minutes)
STAGES = [
    ("weekly", "fast", None, POINT, 6), ("weekly", "fast", "item_id", POOLED, 2),
    ("weekly", "slow", None, POINT, 6), ("weekly", "slow", "item_id", POOLED, 2),
    ("everyday", "fast", None, POINT, 30), ("everyday", "fast", "item_id", POOLED, 8),
    ("everyday", "slow", None, POINT, 30), ("everyday", "slow", "item_id", POOLED, 8),
]


def stamp():
    return datetime.now().strftime("%H:%M:%S")


for suite, item, pool, methods, est in STAGES:
    cmd = [sys.executable, "-m", "src.run", "--suite", suite, "--item", item, "--methods", *methods,
           "--note", NOTE, "--quiet"] + (["--pool", pool] if pool else [])
    print(f"\n[{stamp()}] START {suite} {item} pool={pool} ({len(methods)} methods, est {est} min, limit {2*est} min)", flush=True)
    t = time.time()
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=2 * est * 60)
        print(out.stdout[-3000:], flush=True)
        if out.returncode != 0:
            print(out.stderr[-3000:], flush=True)
            print(f"[{stamp()}] FAILED {suite} {item} pool={pool} after {(time.time()-t)/60:.1f} min", flush=True)
            sys.exit(1)
        print(f"[{stamp()}] DONE {suite} {item} pool={pool} in {(time.time()-t)/60:.1f} min", flush=True)
    except subprocess.TimeoutExpired as e:
        print((e.stdout or b"")[-2000:] if isinstance(e.stdout, bytes) else (e.stdout or "")[-2000:], flush=True)
        print(f"[{stamp()}] TIMEOUT {suite} {item} pool={pool}: more than {2*est} min, killed", flush=True)
        sys.exit(2)
print(f"\n[{stamp()}] ALL STAGES DONE", flush=True)
