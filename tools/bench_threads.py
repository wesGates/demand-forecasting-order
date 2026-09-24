"""Does parallelising across fits beat parallelising inside a fit on this CPU?
Times real XGBoost and ARIMA fits from the study, three ways:
  A. one fit, all cores (today's setting, n_jobs=-1)
  B. one fit, one thread
  C. 20 single-thread fits at once in 20 processes (the proposed harness)
Run from the repo root with PYTHONPATH=. ; takes about 2 min."""
import os, time, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")  # for the workers; the parent resets below
warnings.simplefilter("ignore")
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

from src.step1_problem import Config, STUDY_ITEMS
from src.step2_data import load_panel
from src.step5_evaluate import supervised_matrices
from src.models.base import Context
from src.models import xgboost_model, arima_model

N_WORKERS = 20


def make_contexts():
    cfg = Config(item_ids=STUDY_ITEMS)
    df = load_panel(cfg, verbose=False)
    mats = supervised_matrices(df, cfg)
    origins = cfg.fold_origins(df["date"].max())
    ctxs = []
    for sid, series in df.groupby("id", sort=True, observed=True):
        series = series.sort_values("date").reset_index(drop=True)
        for origin in origins[-2:]:  # two origins x ten stores = 20 fits
            hist = series[series["date"] <= origin]
            targets = series[(series["date"] > origin) & (series["date"] <= origin + pd.Timedelta(days=7))]
            ctxs.append(Context(history=hist, targets=targets.drop(columns=["sales"]), origin=origin,
                                horizon=7, series_id=sid, train_pool=mats[sid], seed=0))
    return ctxs[:N_WORKERS]


def xgb_fit(args):
    ctx, n_jobs = args
    t = time.time()
    xgboost_model.fit_predict_xgboost(ctx, n_jobs=n_jobs)
    return time.time() - t


def arima_fit(ctx):
    t = time.time()
    arima_model.arima_orders.clear()
    arima_model.fit_predict_arima(ctx)
    return time.time() - t


if __name__ == "__main__":
    ctxs = make_contexts()
    print(f"{len(ctxs)} fits, training rows per fit ~{len(ctxs[0].train_pool)}")

    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
    t = time.time(); a = [xgb_fit((c, -1)) for c in ctxs]; tA = time.time() - t
    print(f"A  XGBoost, one at a time, all cores each : {tA:6.1f} s total, {np.mean(a):.2f} s/fit")
    t = time.time(); b = [xgb_fit((c, 1)) for c in ctxs[:5]]; tB = (time.time() - t) * len(ctxs) / 5
    print(f"B  XGBoost, one at a time, one thread each: {tB:6.1f} s total (est. from 5), {np.mean(b):.2f} s/fit")
    os.environ["OMP_NUM_THREADS"] = "1"
    t = time.time()
    with ProcessPoolExecutor(N_WORKERS) as ex:
        c_ = list(ex.map(xgb_fit, [(c, 1) for c in ctxs]))
    tC = time.time() - t
    print(f"C  XGBoost, {N_WORKERS} at once, one thread each: {tC:6.1f} s total, {np.mean(c_):.2f} s/fit -> {tA / tC:.1f}x faster than A")

    t = time.time(); d = [arima_fit(c) for c in ctxs[:4]]; tD = (time.time() - t) * len(ctxs) / 4
    print(f"D  ARIMA, one at a time (order search each): {tD:6.1f} s total (est. from 4), {np.mean(d):.1f} s/fit")
    t = time.time()
    with ProcessPoolExecutor(N_WORKERS) as ex:
        e = list(ex.map(arima_fit, ctxs))
    tE = time.time() - t
    print(f"E  ARIMA, {N_WORKERS} at once                  : {tE:6.1f} s total, {np.mean(e):.1f} s/fit -> {tD / tE:.1f}x faster than D")
