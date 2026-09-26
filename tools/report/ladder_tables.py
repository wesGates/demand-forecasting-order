"""Forecast-value-added ladder and the full method x baseline matrix, from cache.
Run from the exp worktree root with PYTHONPATH=. ; prints markdown tables and
writes report/figures/1_ladder.png (path given as argv[1])."""
import sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from src import plots
from src.step1_problem import Config
from src.step5_evaluate import _read_cached
from src.scoring import score_folds
L = dict(fold_step=1, n_folds=358)
NAME = {"seasonal_naive": "This day last week (seasonal naive)", "arma": "ARMA, no seasonality, no inputs", "arima_plain": "ARIMA, no holiday or SNAP inputs",
        "arima": "ARIMA", "ets": "Exponential smoothing (ETS)", "moving_average_28": "28-day moving average", "xgboost": "XGBoost, one per store",
        "xgboost_rel": "XGBoost, one per store, level-relative", "xgboost@pooled": "XGBoost, pooled", "xgboost_rel@pooled": "XGBoost, pooled, level-relative (final)"}
def load(item, key):
    m, pooled = (key.split("@") + [None])[:2]
    cfg = Config(item_ids=(item,), pool_by="item_id" if pooled else None, **L)
    f = _read_cached(cfg, m); assert f is not None, (item, key); return f
def per_week(f):
    sc = score_folds(f.assign(method="m")); return sc.set_index(["id", "origin"])["rmsse"]
def weekly_mae(f):
    g = f[~f["closure"].astype(bool)].groupby(["id", "origin"]).agg(fc=("forecast", "sum"), ac=("actual", "sum"))
    return float((g.fc - g.ac).abs().mean())
def pair(a, b):
    """b against a, paired on store-week: weeks won, median gain, stores won (per-store RMSSE)."""
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    ok = j.a > 0
    gain = ((j.a - j.b) / j.a * 100)[ok]
    win = float((j.b < j.a).mean())
    ps = j.groupby(level=0).mean()
    return win, float(gain.median()), int((ps.b < ps.a).sum())
def ladder(item, rungs):
    fr = {k: load(item, k) for k in rungs}; pw = {k: per_week(fr[k]) for k in rungs}
    rows = []
    for i, k in enumerate(rungs):
        r = dict(rung=NAME[k], rmsse=f"{float(pw[k].mean()):.2f}", weekly=f"{weekly_mae(fr[k]):.1f}")
        if i:
            w, g, s = pair(pw[rungs[i-1]], pw[k]); r.update(below=f"{w:.0%} / {g:.0f}%".replace("-0%", "0%"), stores=s)
            w, g, _ = pair(pw[rungs[0]], pw[k]); r.update(bottom=f"{w:.0%} / {g:.0f}%")
        else:
            r.update(below="", stores="", bottom="")
        rows.append(r)
    return pd.DataFrame(rows), pw
def matrix(item, methods, baselines, pw_all):
    rows = []
    for k in methods:
        r = {"method": NAME[k]}
        for b in baselines:
            short = {"seasonal_naive": "seasonal naive", "arma": "ARMA", "arima": "ARIMA", "moving_average_28": "28-day average"}[b]
            if b == k:
                r[f"weeks won vs: {short}"] = ""; r[f"median gain vs: {short}"] = ""; continue
            w, g, _ = pair(pw_all[b], pw_all[k])
            r[f"weeks won vs: {short}"] = f"{w:.0%}"; r[f"median gain vs: {short}"] = f"{g:+.0f}%".replace("+0%", "0%").replace("-0%", "0%")
        rows.append(r)
    return pd.DataFrame(rows)
def md(df):
    cols = list(df.columns); out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows(): out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out)

fast_rungs = ["seasonal_naive", "arma", "arima_plain", "arima", "xgboost_rel@pooled"]
slow_rungs = ["seasonal_naive", "arima", "moving_average_28", "xgboost_rel@pooled"]
lf, pwf = ladder("FOODS_3_586", fast_rungs); ls_, pws = ladder("FOODS_1_021", slow_rungs)
hdr = {"rung": "rung", "rmsse": "RMSSE", "below": "vs the rung below: weeks won / median gain", "bottom": "vs seasonal naive: weeks won / median gain", "stores": "stores won of 10 vs the rung below", "weekly": "weekly error, units"}
print("## LADDER fast\n" + md(lf.rename(columns=hdr)) + "\n\n## LADDER slow\n" + md(ls_.rename(columns=hdr)))
all_fast = ["xgboost_rel@pooled", "xgboost@pooled", "xgboost_rel", "arima", "arima_plain", "xgboost", "ets", "arma", "moving_average_28", "seasonal_naive"]
all_slow = ["moving_average_28", "ets", "xgboost_rel@pooled", "xgboost_rel", "arima_plain", "arma", "arima", "xgboost@pooled", "xgboost", "seasonal_naive"]
pwf_all = {k: pwf.get(k) if k in pwf else per_week(load("FOODS_3_586", k)) for k in all_fast}
pws_all = {k: pws.get(k) if k in pws else per_week(load("FOODS_1_021", k)) for k in all_slow}
print("\n## MATRIX fast\n" + md(matrix("FOODS_3_586", all_fast, ["seasonal_naive", "arma", "arima"], pwf_all)))
print("\n## MATRIX slow\n" + md(matrix("FOODS_1_021", all_slow, ["seasonal_naive", "moving_average_28", "arima"], pws_all)))

# Figure: the ladder, both items.
plots.use_style(); plt.rcParams["savefig.dpi"] = 300  # crisp in the PDF
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw=dict(width_ratios=[5, 4]))
for ax, (df, title) in zip(axes, ((lf, "Fast mover"), (ls_, "Slow mover in decline"))):
    y = np.arange(len(df))[::-1]
    cols = [plots.SERIES_5 if "final" in r else plots.INK_SOFT for r in df["rung"]]
    ax.barh(y, df["rmsse"].astype(float), color=cols, height=0.6)
    for yi, (_, r) in zip(y, df.iterrows()):
        ax.text(float(r["rmsse"]) + 0.01, yi, r["rmsse"] + (f"   ({r['below'].split(' / ')[1]} gain over the previous rung)" if r["below"] else ""), va="center", fontsize=8.5)
    ax.set_yticks(y, [r.replace(" (seasonal naive)", "").replace(" (final)", "") for r in df["rung"]], fontsize=8.5)
    ax.set_xlim(0, 1.05); ax.set_xlabel("scaled error (RMSSE)", fontsize=9); ax.set_title(title, fontsize=10); ax.grid(axis="y", visible=False)
fig.suptitle("What each modelling step gained, from the simplest forecast to the final model", fontsize=11)
fig.savefig(sys.argv[1], bbox_inches="tight"); plt.close(fig); print("\nfigure written", sys.argv[1])
