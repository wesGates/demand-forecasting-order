"""Pair every run recorded under one note with the earlier run of the same
method and config (optionally restricted to another note) and print the
registry's paired comparison for each. Expect: benchmarks/ETS/ARIMA identical
(win rate 0.5 by convention, 0.0% everywhere); XGBoost variants near 50% and
median near zero. Run from the repo root: PYTHONPATH=. python tools/compare_runs.py "<new note>" ["<old note>"]"""
import sqlite3
import pandas as pd
from src.registry import compare

con = sqlite3.connect("cache/registry.sqlite")
runs = pd.read_sql_query("SELECT * FROM run", con)
con.close()
import sys
NEW = sys.argv[1] if len(sys.argv) > 1 else "item 7 refit"  # note prefix of the new runs
OLD = sys.argv[2] if len(sys.argv) > 2 else None  # note prefix of the old runs (default: any earlier run)
new = runs[(runs["note"].fillna("").str.startswith(NEW)) & (runs["code_current"] == 1)]
old = runs[~runs["run_id"].isin(new["run_id"])]  # any earlier run of the same config, whatever its code
if OLD:
    old = old[old["note"].fillna("").str.startswith(OLD)]
keys = ["method", "item_ids", "store_ids", "pool_by", "fold_step", "n_folds"]
rows = []
for _, r in new.iterrows():
    match = old
    for k in keys:
        match = match[match[k].fillna("") == (r[k] if pd.notna(r[k]) else "")]
    if match.empty:
        rows.append({"method": r["method"], "suite": r["suite"], "item": r["item_ids"], "pool": r["pool_by"],
                     "verdict": "no old run"}); continue
    prev = match.sort_values("recorded_at").iloc[-1]
    c = compare(prev["run_id"], r["run_id"])
    ident = c["impr_q1_pct"] == 0 and c["impr_q3_pct"] == 0 and abs(c["rmsse_a"] - c["rmsse_b"]) < 1e-12
    rows.append({"method": r["method"], "suite": r["suite"], "item": r["item_ids"], "pool": r["pool_by"],
                 "rmsse_old": round(c["rmsse_a"], 4), "rmsse_new": round(c["rmsse_b"], 4),
                 "win_new": round(c["win_rate_b"], 2), "median_pct": round(c["impr_median_pct"], 2),
                 "q1_pct": round(c["impr_q1_pct"], 2), "q3_pct": round(c["impr_q3_pct"], 2),
                 "verdict": "IDENTICAL" if ident else ("ok" if abs(c["impr_median_pct"]) < 2 and 0.35 <= c["win_rate_b"] <= 0.65 else "CHECK")})
pd.set_option("display.width", 220)
out = pd.DataFrame(rows).sort_values(["suite", "item", "pool", "method"])
print(out.to_string(index=False))
print("\nflagged:", int((out["verdict"] == "CHECK").sum()), " identical:", int((out["verdict"] == "IDENTICAL").sum()),
      " ok:", int((out["verdict"] == "ok").sum()), " no old run:", int((out["verdict"] == "no old run").sum()))
