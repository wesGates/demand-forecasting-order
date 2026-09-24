"""Does the current code reproduce the last cached run of each method?
For each config and method: fit from scratch with the new harness (all cores)
and compare with the last cached file made for the same config (found by its
sidecar, whatever code digest it carries). Benchmarks, ETS and ARIMA must be
identical; XGBoost may differ in the last decimals (one thread per fit now)
and is reported as max |diff| and the RMSSE change. Also times each method.
Run from the repo root with PYTHONPATH=. ."""
import warnings, time, json, sys; warnings.simplefilter("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from src.step1_problem import Config, STUDY_ITEMS, DEV
from src.step2_data import load_panel
from src.step5_evaluate import run_walk_forward, _config_fields
from src.scoring import score_folds

EXACT = ["mean", "naive", "seasonal_naive", "seasonal_naive_364", "drift", "moving_average_28", "ets", "arima"]
TREES = ["xgboost", "xgboost_rel"]


def old_cached(cfg, method):
    want = _config_fields(cfg)
    hits = []
    for meta in Path("cache/predictions").glob(f"{method}_*.json"):
        info = json.loads(meta.read_text())
        if info["method"] == method and info["config"] == want:
            hits.append(meta)
    if not hits:
        return None
    meta = max(hits, key=lambda m: m.stat().st_mtime)
    return pd.read_parquet(meta.with_suffix(".parquet"))


def compare(old, new):
    key = ["id", "fold", "method", "horizon"]
    a = old.sort_values(key).reset_index(drop=True)
    b = new.sort_values(key).reset_index(drop=True)
    if len(a) != len(b) or (a[key].values != b[key].values).any():
        return "ROW MISMATCH", np.nan, np.nan
    diff = float(np.abs(a["forecast"].to_numpy() - b["forecast"].to_numpy()).max())
    ra = score_folds(a)["rmsse"].mean(); rb = score_folds(b)["rmsse"].mean()
    return ("IDENTICAL" if diff == 0 else "differs"), diff, rb - ra


ok = True
for label, cfg in (("DEV per-store", Config(**DEV)),
                   ("DEV pooled", Config(**DEV, pool_by="item_id")),
                   ("weekly per-store", Config(item_ids=STUDY_ITEMS)),
                   ("weekly pooled", Config(item_ids=STUDY_ITEMS, pool_by="item_id"))):
    df = load_panel(cfg, verbose=False)
    methods = TREES if cfg.pool_by else EXACT + TREES
    print(f"\n=== {label}", flush=True)
    for m in methods:
        old = old_cached(cfg, m)
        t = time.time()
        new = run_walk_forward(df, cfg, methods=[m], progress=False, use_cache=False)
        secs = time.time() - t
        if old is None:
            print(f"  {m:<20} {secs:7.1f} s   (no old cache to compare)", flush=True); continue
        verdict, diff, drmsse = compare(old, new)
        exact_required = m in EXACT
        if exact_required and verdict != "IDENTICAL":
            ok = False; verdict = "FAIL " + verdict
        print(f"  {m:<20} {secs:7.1f} s   {verdict:<14} max|diff|={diff:.3g}  dRMSSE={drmsse:+.4f}", flush=True)
print("\nGATE 3:", "PASSED" if ok else "FAILED", flush=True)
sys.exit(0 if ok else 1)
