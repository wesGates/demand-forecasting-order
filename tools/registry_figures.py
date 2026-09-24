"""Export the registry to files a person can read, and draw figures from it.
Run from the repo root with PYTHONPATH=. . Writes to figures/registry/ :
  runs.csv                 every run row (254)
  runs_current.md          the 48 current runs, one table per suite and item, with shorthand
  fold_scores_current.csv  every store-origin score of the current runs
  figures/1_rmsse_by_run_<suite>.png   RMSSE and bias per run, shorthand labels + key
  figures/2_progression.png            stage by stage: per-store -> pooled -> level-relative
  figures/3_speed.png                  old vs new run times per stage
"""
import sqlite3
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from src import plots

OUT = Path("figures/registry")  # gitignored, like every rendered figure
(OUT / "figures").mkdir(parents=True, exist_ok=True)
plots.use_style()

con = sqlite3.connect("cache/registry.sqlite")
runs = pd.read_sql_query("SELECT * FROM run ORDER BY suite, item_ids, pool_by, rmsse_mean", con)
cur = runs[runs["code_current"] == 1].copy()
ids = ",".join(f"'{r}'" for r in cur["run_id"])
folds = pd.read_sql_query(f"SELECT * FROM fold_score WHERE run_id IN ({ids})", con)
con.close()

# ---- shorthand -----------------------------------------------------------
SHORT = {"xgboost": "XGB", "xgboost_rel": "XGB·R", "arima": "ARIMA", "ets": "ETS",
         "moving_average_28": "MA28", "seasonal_naive": "SN7", "seasonal_naive_364": "SN364",
         "mean": "MEAN", "naive": "NAIVE", "drift": "DRIFT"}
KEY = [("XGB", "gradient-boosted trees, plain target"),
       ("XGB·R", "trees on the level-relative target (sales − 28-day mean at origin, added back)"),
       ("·P", "pooled: one model per origin for all ten stores, store id as a feature"),
       ("(no ·P)", "one model per store"),
       ("ARIMA", "seasonal ARIMA (d=0, D=1, s=7) with holiday, run-up and SNAP regressors"),
       ("ETS", "Holt-Winters exponential smoothing, weekly season"),
       ("MA28", "28-day moving average"), ("SN7", "seasonal naive: this day last week (reference)"),
       ("SN364", "this day last year"), ("MEAN / NAIVE / DRIFT", "whole-history mean; last value; last value plus trend"),
       ("weekly", "52 origins, one per week (development layout)"),
       ("everyday", "358 origins, every day (reported layout)"),
       ("fast", "FOODS_3_586, 14–103 units/day"), ("slow", "FOODS_1_021, 0.6–7.2 units/day, declining")]
ITEM = {"FOODS_3_586": "fast", "FOODS_1_021": "slow"}


def label(r):
    return SHORT.get(r["method"], r["method"]) + ("·P" if isinstance(r["pool_by"], str) else "")


cur["label"] = cur.apply(label, axis=1)
cur["item"] = cur["item_ids"].map(ITEM)

# ---- exports -------------------------------------------------------------
runs.to_csv(OUT / "runs.csv", index=False)
folds.merge(cur[["run_id", "label", "suite", "item"]], on="run_id").to_csv(OUT / "fold_scores_current.csv", index=False)
cols = ["label", "run_id", "rmsse_mean", "rmsse_normal", "rmsse_holiday", "bias_mean", "win_vs_ref", "impr_q1", "impr_median", "impr_q3", "win_vs_stress", "run_seconds", "git_commit", "recorded_at"]
with open(OUT / "runs_current.md", "w") as f:
    f.write("# Current runs (made under the parallel harness, 2026-09-23)\n\n")
    f.write("Shorthand: " + "; ".join(f"**{k}** = {v}" for k, v in KEY) + "\n\n")
    for (suite, item), g in cur.groupby(["suite", "item"]):
        f.write(f"## {suite}, {item} item\n\n")
        f.write(g.sort_values("rmsse_mean")[cols].round(3).to_markdown(index=False) + "\n\n")

# ---- figure 1: RMSSE and bias per run, one panel per item, per suite --------
for suite in ("everyday", "weekly"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
    for ax, item in zip(axes, ("fast", "slow")):
        g = cur[(cur["suite"] == suite) & (cur["item"] == item)].sort_values("rmsse_mean", ascending=False)
        y = np.arange(len(g))
        colors = [plots.SERIES_1 if "XGB" in l and "R" not in l else plots.SERIES_5 if "XGB·R" in l
                  else plots.SERIES_3 if l == "ARIMA" else plots.SERIES_2 if l == "ETS" else plots.INK_SOFT for l in g["label"]]
        ax.hlines(y, 0, g["rmsse_mean"], color=colors, lw=1.2, alpha=0.6)
        ax.scatter(g["rmsse_mean"], y, color=colors, s=42, zorder=3)
        for yi, (_, r) in zip(y, g.iterrows()):
            ax.text(r["rmsse_mean"] + 0.012, yi, f"{r['rmsse_mean']:.3f}  bias {r['bias_mean']:+.2f}", va="center", fontsize=8, color=plots.INK_SOFT)
        ax.set_yticks(y, g["label"]); ax.axvline(1.0, color=plots.INK_MUTED, lw=1)
        ax.set_xlim(0.3, max(1.15, g["rmsse_mean"].max() + 0.25)); ax.set_xlabel("mean RMSSE (1.0 = seasonal naive on training data)")
        ax.set_title(f"{item} item ({[k for k, v in ITEM.items() if v == item][0]}), {suite} layout")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Every current run in the registry: scaled error and bias (units/day)", fontsize=11)
    fig.savefig(OUT / "figures" / f"1_rmsse_by_run_{suite}.png", bbox_inches="tight"); plt.close(fig)

# key figure: the shorthand, as a picture that can sit beside any of the others
fig, ax = plt.subplots(figsize=(9, 4.2)); ax.axis("off")
for i, (k, v) in enumerate(KEY):
    ax.text(0.01, 1 - i * 0.068, k, fontsize=9, fontweight="bold", va="top", transform=ax.transAxes)
    ax.text(0.24, 1 - i * 0.068, v, fontsize=9, va="top", transform=ax.transAxes)
ax.set_title("Key to the shorthand", fontsize=11, loc="left")
fig.savefig(OUT / "figures" / "0_key.png", bbox_inches="tight"); plt.close(fig)

# ---- figure 2: progression, every-day layout ---------------------------------
stages = [("item 2\nper store", "XGB"), ("item 3\npooled", "XGB·P"), ("item 5\npooled +\nlevel-relative", "XGB·R·P")]
refs = [("ARIMA", plots.SERIES_3), ("ETS", plots.SERIES_2), ("MA28", plots.INK_SOFT), ("SN7", plots.INK_MUTED)]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for j, (metric, title) in enumerate((("rmsse_mean", "mean RMSSE (lower is better)"), ("win_vs_ref", "share of forecasts beating SN7"), ("bias_mean", "bias, units/day (0 is unbiased)"))):
    ax = axes[j]
    for item, color, marker in (("fast", plots.SERIES_1, "o"), ("slow", plots.SERIES_5, "s")):
        g = cur[(cur["suite"] == "everyday") & (cur["item"] == item)].set_index("label")
        ys = [g.loc[l, metric] for _, l in stages]
        ax.plot(range(3), ys, marker=marker, color=color, lw=2, label=f"{item} item, learned model")
        for name, c in refs:
            if name in g.index and metric != "win_vs_ref" or (name in g.index and metric == "win_vs_ref" and name != "SN7"):
                ax.hlines(g.loc[name, metric], -0.15, 2.15, color=c, lw=1, ls="--" if item == "slow" else "-", alpha=0.8,
                          label=f"{item}: {name}" if j == 0 else None)
    ax.set_xticks(range(3), [s for s, _ in stages], fontsize=8); ax.set_title(title, fontsize=10)
    if metric == "bias_mean": ax.axhline(0, color=plots.INK, lw=1)
fig.suptitle("The learned model, stage by stage, on the every-day layout (references as horizontal lines)", fontsize=11)
plots.legend_below(axes[0], ncols=5)
fig.savefig(OUT / "figures" / "2_progression.png", bbox_inches="tight"); plt.close(fig)

# ---- figure 3: speed, old serial harness vs new --------------------------------
# old times from the item 2-5 findings (stage totals); new from the registry's run_seconds
old = {("everyday", "fast", None): 131, ("everyday", "slow", None): 98, ("everyday", "fast", "item_id"): 16, ("everyday", "slow", "item_id"): 15,
       ("weekly", "fast", None): 21, ("weekly", "slow", None): 21, ("weekly", "fast", "item_id"): 4.3, ("weekly", "slow", "item_id"): 1.9}
cur["pool_key"] = cur["pool_by"].where(cur["pool_by"].notna(), None).map(lambda v: v if isinstance(v, str) else "")
new = cur.groupby(["suite", "item", "pool_key"])["run_seconds"].sum() / 60
labels, o, n = [], [], []
for (suite, item, pool), minutes in old.items():
    labels.append(f"{suite}\n{item}{' pooled' if pool else ''}"); o.append(minutes); n.append(float(new.get((suite, item, pool or ""), np.nan)))
fig, ax = plt.subplots(figsize=(11, 3.8)); x = np.arange(len(labels))
ax.bar(x - 0.2, o, 0.4, color=plots.INK_SOFT, label="serial harness (items 2–5, from the findings)")
ax.bar(x + 0.2, n, 0.4, color=plots.SERIES_1, label="parallel harness (item 7 refit, from the registry)")
for xi, (a, b) in enumerate(zip(o, n)):
    ax.text(xi - 0.2, a, f"{a:.0f}", ha="center", va="bottom", fontsize=8); ax.text(xi + 0.2, b, f"{b:.1f}", ha="center", va="bottom", fontsize=8, color=plots.SERIES_1)
ax.set_xticks(x, labels, fontsize=8); ax.set_ylabel("minutes per stage"); ax.set_yscale("log"); ax.set_title("Run time per stage, old harness against new (log scale)", fontsize=11)
plots.legend_below(ax, ncols=2)
fig.savefig(OUT / "figures" / "3_speed.png", bbox_inches="tight"); plt.close(fig)
print("written to", OUT)
