# Demand forecasting on M5

The first study, tag `first-study` in this repository, compared XGBoost,
ETS and seasonal ARIMA against six benchmarks on one fast-moving grocery
item at ten stores, on 52 weekly folds. Everything after it takes the same
pipeline further, one numbered item at a time, each on its own branch
with its own findings file. The report is `report/report.md` and the
two-page summary `report/brief.md`. (An archived copy of the first study
as its own repository is `demand-forecasting-first-study`.)

| item | question | answer |
|---|---|---|
| 1 | Does XGBoost trained on the quantile objective beat calibrated error quantiles? | No. Same pinball loss within 0.002, at 26× the fit cost. |
| 2 | Does making every day an origin change the results? | Tables move by less than 0.005. Two benchmarks lose a Sunday artefact. Adopted as the reported layout. |
| 3 | One model pooled across the ten stores? | Best method on the fast mover. Helps the quiet stores most and cuts bias by two thirds. |
| 4 | An intermittent, declining item? | The learned models lose at every store. The cause is level drift, which trees cannot extrapolate. |
| 5 | Trees on a level-relative target? | Closes the gap on the declining item, small gain on the fast mover. |
| 6 | A run registry | Every run on record with its config, code version, commit and scores, and paired comparisons between runs. |
| 7 | A parallel harness | Every-day runs in 15 min per item, down from about 2 h. Forecasts identical. |
| 8 | A code review | One leak fixed (the closure imputation looked forward), fallbacks and unscored folds counted, the cache checked against the data. |

Findings files are in `findings/`, dated, one per item. `PLAN.md` has the
status, what comes next and the rules. `GLOSSARY.md` defines the terms.

## Results

Every-day layout, 358 origins, ten stores, the year 25 May 2015 to 22 May
2016. RMSSE is the error scaled by the seasonal naïve on training data, so
1.0 means no better than "this day last week". Bias is forecast minus
actual in units a day.

| method | fast mover (FOODS_3_586) | bias | slow, declining item (FOODS_1_021) | bias |
|---|---|---|---|---|
| XGBoost, pooled, level-relative target | **0.611** | +0.21 | 0.501 | +0.18 |
| XGBoost, pooled | 0.622 | +0.26 | 0.570 | +0.57 |
| ARIMA, seasonal, holiday regressors | 0.648 | +0.00 | 0.519 | +0.33 |
| XGBoost, one model per store | 0.665 | +0.76 | 0.610 | +0.95 |
| ETS, Holt-Winters | 0.670 | −0.06 | 0.500 | +0.14 |
| 28-day moving average | 0.795 | −0.03 | **0.497** | +0.08 |
| seasonal naïve, this day last week | 0.864 | −0.04 | 0.687 | +0.03 |

On the fast mover the pooled level-relative XGBoost beats the seasonal
naïve in 84% of seven-day forecasts, by a median of 28%, and is the best
method at seven stores; the plain pooled model takes the other three. On
the slow item the simple methods lead and the same model ties them. The
per-store trees that started the study were 60% high on it.

## How the pipeline works

- `src/step2_data.py` loads M5, reshapes it to one row per store, item and
  day, joins the calendar, SNAP and price data, and fills the Christmas
  closure from the four preceding weeks (FPP §13.7 for the closure as a
  missing day, §5.10 for why only the past is used).
- `src/step3_explore.py` screens each series for stock-out gaps and
  classifies demand by ADI and CV² on training data only.
- `src/features.py` builds lags, rolling summaries, same-weekday means and
  holiday proximity from data at or before the origin, with an assertion
  on every fold.
- `src/models/` holds one module per forecaster. `src/step4_models.py`
  registers them.
- `src/step5_evaluate.py` runs the walk-forward in parallel and caches each
  method's forecasts under a key made of the config and the code the
  method depends on. `src/scoring.py` turns forecasts into tables.
- `src/registry.py` records every run. `src/suites.py` names the layouts.
  `src/run.py` runs, registers and compares.
- `src/validate.py` is the gate: ten checks, including negative controls
  on synthetic data.

## Running it

```
python -m venv .venv                       # Python 3.14
source .venv/bin/activate
pip install -r requirements.txt
```

Put the Kaggle M5 files `sales_train_evaluation.csv`, `calendar.csv` and
`sell_prices.csv` in `data/` (gitignored). Then:

```
python -m src.validate                     # ten checks; run before quoting a number
python -m pytest tests                     # the test suite
python -m src.run --suite dev --item fast --methods xgboost_rel --note "what this tries"
python -m src.registry list everyday       # every run on record, on the reported layout
python -m src.registry last <run_id>       # a run against its predecessor
```

Suites fix the layout a change is scored on. `dev` is three stores and
eight weekly folds (under a minute), `weekly` is 52 folds (about 3 min for
every method), `everyday` is 358 origins (about 15 min). Develop on `dev`
and confirm on `everyday`.

`tools/` has the refit, gate, comparison and figure scripts, each described
in `tools/README.md`.

## The cache and the registry

Each method's forecasts are cached under `cache/predictions/` with a key
made of the config (paths excluded) and a digest of the code the method
depends on. Editing one model refits that model only. A cached run is
checked against the data on disk every time it is read. The registry
(`cache/registry.sqlite`) lists every run with its config, code digest,
commit, time and scores, plus one row per store-origin, and can be rebuilt
from the cache with `python -m src.registry backfill`. Both live under
`cache/`, which is gitignored.

## Layout

```
src/
  step1_problem.py    Config, every decision in one frozen object
  step2_data.py       load, join, trim, impute closures, cache the panel
  step3_explore.py    availability screen, demand classification, event effects
  features.py         origin-based features and the leakage assertion
  step4_models.py     the forecaster registry
  models/             one module per forecaster
  step5_evaluate.py   the parallel walk-forward harness and the cache
  scoring.py          fold scores, tables, win rates, improvement quartiles
  registry.py         the run registry
  suites.py           the named layouts
  run.py              run, register, compare
  order.py            the order-quantity prototype (calibrated quantiles)
  plots.py            every figure, one palette
  render_figures.py   figures for the notebooks
  validate.py         the gate
tools/                refit, gate, comparison and figure scripts
tests/                pytest suite
notebooks/            01_explore, 02_evaluate, 03_order (percent-format scripts)
findings/             one dated file per item, with provenance
report/               the write-up and its figures
```

## What is not claimed

Public data only. M5 has no inventory, receipts or stock-outs, so a
business-outcome backtest is not possible and no store-level result is
implied. Two items and ten stores. New items with no history are not
addressed. Deep learning, cloud and SQL are not used; the plan lists what
comes next.
