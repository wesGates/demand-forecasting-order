"""Forecast-level comparison between two refits (by registry note):
for each method and config, the share of forecasts that are bit-identical,
the max absolute difference, where the differences sit in the calendar,
and how much the RMSSE scale moved per series. Reads the cache only.
Run from the repo root: PYTHONPATH=. python tools/compare_forecasts.py ["<new note>" "<old note>"]"""
import json, sqlite3
from pathlib import Path
import numpy as np, pandas as pd

con = sqlite3.connect("cache/registry.sqlite")
runs = pd.read_sql_query("SELECT run_id, method, config_json, note, cache_file FROM run", con)
con.close()
import sys
NEW = sys.argv[1] if len(sys.argv) > 1 else "item 8 refit"
OLD = sys.argv[2] if len(sys.argv) > 2 else "item 7 refit"
new = runs[runs["note"].fillna("").str.startswith(NEW)]
old = runs[runs["note"].fillna("").str.startswith(OLD)]
pd.set_option("display.width", 220)
rows = []
scales = []
for _, r in new.iterrows():
    def same_config(a, b):
        """Equal on every field both configs have; a field removed since (mask_holidays) does not count."""
        ja, jb = json.loads(a), json.loads(b)
        return all(ja[k] == jb[k] for k in ja.keys() & jb.keys())

    match = old[(old["method"] == r["method"]) & old["config_json"].map(lambda c: same_config(c, r["config_json"]))]
    if match.empty:
        continue
    a = pd.read_parquet(Path("cache/predictions") / match.iloc[0]["cache_file"])
    b = pd.read_parquet(Path("cache/predictions") / r["cache_file"])
    key = ["id", "fold", "horizon"]
    m = a.merge(b, on=key, suffixes=("_7", "_8"))
    same = np.isclose(m["forecast_7"], m["forecast_8"], rtol=0, atol=1e-9)
    diff_dates = pd.to_datetime(m.loc[~same, "target_date_8"])
    cfg = json.loads(r["config_json"])
    rows.append({"method": r["method"], "layout": "everyday" if cfg["fold_step"] == "1" else "weekly",
                 "item": cfg["item_ids"].strip("(),'"), "pool": cfg["pool_by"].strip("'"),
                 "forecasts": len(m), "identical_share": round(float(same.mean()), 4),
                 "max_abs_diff": round(float(np.abs(m["forecast_7"] - m["forecast_8"]).max()), 3),
                 "diff_dates": "" if same.all() else f"{diff_dates.min().date()} .. {diff_dates.max().date()}"})
    if r["method"] == "seasonal_naive" and not cfg["pool_by"].strip("'") != "None":
        pass
    if r["method"] == "seasonal_naive":
        s = m.groupby("id")[["scale_7", "scale_8"]].first()
        s["change_pct"] = (s["scale_8"] / s["scale_7"] - 1) * 100
        s["layout"] = rows[-1]["layout"]
        scales.append(s)
out = pd.DataFrame(rows).sort_values(["layout", "item", "pool", "method"])
print(out.to_string(index=False))
print("\nRMSSE scale per series, item 7 -> item 8 (% change; the pre-holdout Christmases were re-imputed):")
print(pd.concat(scales).round(3).to_string())
