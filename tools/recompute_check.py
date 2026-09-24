"""Recompute two cached runs from scratch and compare them to the cache, row by row.
Weekly layout (52 folds) so it takes ~6 min: per-store XGBoost (~4 min) and
pooled XGBoost (~2 min), FOODS_3_586. Run from the repo root with PYTHONPATH=.
Prints IDENTICAL or the largest difference."""
import warnings, time; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from src.step1_problem import Config, STUDY_ITEMS
from src.step2_data import load_panel
from src.step5_evaluate import run_walk_forward, _read_cached

for pool in (None, "item_id"):
    cfg = Config(item_ids=STUDY_ITEMS, pool_by=pool)
    df = load_panel(cfg, verbose=False)
    cached = _read_cached(cfg, "xgboost")
    assert cached is not None, "no cached run for this config"
    t = time.time()
    fresh = run_walk_forward(df, cfg, methods=["xgboost"], progress=False, use_cache=False)
    key = ["id", "fold", "method", "horizon"]
    a = cached.sort_values(key).reset_index(drop=True)
    b = fresh.sort_values(key).reset_index(drop=True)
    same_rows = len(a) == len(b) and (a[key].values == b[key].values).all()
    diff = float(np.abs(a["forecast"].to_numpy() - b["forecast"].to_numpy()).max()) if same_rows else float("nan")
    print(f"pool={pool}: {len(b)} rows, {time.time()-t:.0f} s, "
          + ("IDENTICAL" if same_rows and diff == 0 else f"max |difference| = {diff}"), flush=True)
