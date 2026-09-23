# Plan and handoff

Written 2026-09-22, when this repository was split from
[demand-forecasting](https://github.com/wesGates/demand-forecasting). Read
this first. It says what the parent established, what is here, what to do
next and in what order, and the rules of the road.

## What the parent established

One fast-moving grocery item (`FOODS_3_586`) at ten stores, 52 walk-forward
folds of 7 days over 25 May 2015 – 22 May 2016, origins 7 days apart (all
Sundays). Against the seasonal naïve reference (this day last week):

| model | win rate | median improvement | Q1–Q3 | mean RMSSE (all / normal / holiday) | bias, units/day |
|---|---|---|---|---|---|
| ARIMA (seasonal, holiday regressors) | 82% | 24% | 10–37% | 0.65 / 0.63 / 0.74 | −0.1 |
| ETS (Holt-Winters) | 83% | 21% | 8–33% | 0.67 / 0.64 / 0.80 | −0.1 |
| XGBoost | 75% | 23% | 0–40% | 0.67 / 0.65 / 0.72 | +0.9 |

XGBoost wins at the Texas stores and on holiday weeks; the classical models
at the California stores and on ordinary weeks. XGBoost over-forecasts by
about 4% at the busiest store. Full write-up: the parent's `report/`.

## What is here that the parent removed

- `src/order.py`: weekly totals, calibrated quantile forecasts (point
  forecast plus the τ-quantile of a method's own errors over a calibration
  year), pinball loss with FPP's factor of two, coverage, expanding-window
  variant.
- `notebooks/03_order.py`, `tests/test_order.py`, validator check 7, the
  order figures.
- Prototype result (calibrated quantiles, two-year run, all on step 7):
  ARIMA best pinball at τ 0.3/0.5/0.7, ties XGBoost at 0.9; XGBoost fifth at
  τ 0.3 because its errors shrink year over year so a fixed calibration
  year overstates its spread; coverage 0.89–0.92 at τ 0.9 for every model.

## The work, in order

1. **XGBoost on the quantile objective.** A new method that fits one
   multi-quantile model per store and origin (`reg:quantileerror`, several
   τ at once) and exposes one registered forecaster per τ. Compare with ETS
   and ARIMA given the same treatment: their daily τ-quantiles from
   calibrated error quantiles per horizon. Score daily pinball by τ and
   coverage; then weekly totals. Acceptance: a table of pinball by method
   and τ on the same folds, with coverage, where "same treatment" is stated
   exactly.
2. **Every day an origin.** `Config(fold_step=1, n_folds=358)` scores the
   same year with origins rotating through the week. One run of about
   1 h 45 min for all methods; from then on it is the reported layout and
   step 7 is for development. Acceptance: the parent's tables reproduced on
   the new layout, with the horizon plot clean and one-step residuals.
3. **Pooled across stores.** `Config(pool_by="item_id")`, one model for the
   item over all stores with store identity as a feature. This is the
   design that scales to thousands of items. Acceptance: by-store RMSSE
   against the per-store models, especially the quiet stores.
4. **An intermittent item.** Choose by the demand classification (ADI ≥ 1.32
   or CV² ≥ 0.49 at some stores). Acceptance: the same tables, with the
   classical and simple methods expected to win where the earlier analysis
   said they would.

## Status, 2026-09-23

| item | branch | outcome |
|---|---|---|
| 1. quantile objective | on main | matches post-hoc calibration within 0.002 at 26x the cost; ARIMA + calibration best |
| 2. every day an origin | `item2-fold-step-1`, merged | tables within 0.005; naive/drift Sunday bias was an artefact; adopted as reported layout |
| 3. pooled XGBoost | `item3-pooling`, merged | best method on the fast mover: 0.621 vs ARIMA 0.647; Q1 improvement 10% vs 1% |
| 4. intermittent item | `item4-intermittent` | learned models lose at every store; cause is level drift trees cannot extrapolate |
| 5. level-relative target | `item5-level-relative-target` | development numbers only (weekly origins): pooled 0.498 vs moving average 0.496 on the intermittent item (was 0.571); 0.616, best, on the fast mover; every-day confirmation not run |

Next, in order: (a) a level-relative target for the trees on declining
items; (b) the pooled model's own residuals and calibrated quantiles; (c)
pooling wider than the item; (d) a parallel harness over stores and folds
so full runs take minutes on a large machine; (e) the second report, from
the every-day layout, with items 3 and 4 as its two results; (f) the
Fourier/STL item below.

## Later, not yet scheduled

- **Fourier terms and STL decomposition** (FPP §13.1). Daily data carries a
  weekly and an annual cycle; the classical models here take only the
  weekly one. Two routes: Fourier terms for the annual cycle as regressors
  in the ARIMA (dynamic harmonic regression), or an STL decomposition with
  a model on the seasonally adjusted series. Worth evaluating once the four
  items above are done, since the annual shape is where the classical
  models could gain against XGBoost, which already sees day of year.
- **A run registry, so every change is measured against the last one.**
  An append-only table, one row per cached run: run id (the cache
  digest), method, config fields, code digest, branch, commit, time,
  run time, and the headline scores on a fixed benchmark suite (`DEV`,
  weekly, every-day; later a class-stratified item set). A comparison
  script reports a change against its predecessor paired by store and
  origin (win rate and improvement quartiles, as the reports already do),
  so a difference inside the fold-to-fold noise is not read as progress.
  File-based first (parquet); the same schema becomes the SQL `run` table.
- **Faster iteration.** A parallel harness over stores and origins
  (processes, one XGBoost thread each; one writer per cache file; pooled
  fits memoised per origin; ARIMA orders chosen once and shared). Every
  full run from hours to minutes on this machine.
- **Intermittent demand, properly.** A class-stratified item sample,
  Croston and TSB as registered methods, results by class, and the deep
  learning models FPP covers, compared on the same folds.
- **New items (zero-shot / cold start).** Needs its own evaluation design:
  the harness requires a year of history (`min_train_days`), so new items
  are scored on items launched inside the test period, against simple
  fallbacks (category or store averages) and pretrained forecasting models
  that need no history of the item.
- **SQL Server as the results store** (designed 2026-09-23, not built).
  SQL Server in a Docker container as source and sink for the pipeline,
  in new files only, so no cached run is invalidated:
  1. sales, calendar, store and item tables loaded from the M5 files, with
     a test that the panel read back equals `load_panel`;
  2. a `run` table with one row per cache entry (run id = cache digest;
     method, `pool_by`, `fold_step`, `n_folds`, horizon, seed, item and
     store scope, code digest, branch, commit, full config as JSON) and a
     `forecast` table keyed by run, store, item, origin and target date;
     actuals come from a join on the sales table, not a copy; filled by an
     idempotent sync from `cache/predictions/`, never edited by hand;
  3. scoring views (RMSSE, bias, WAPE, MAPE, win rate, quartiles by store
     and kind of week) with a test that they equal the pandas tables;
  4. a daily job that reads history up to a date, fits, and writes the next
     seven days; rerunning a date must leave the table unchanged.
  Optional throughout: everything still runs without a database. CI can
  run the tests against a SQL Server service container on synthetic
  series, never the licensed data. Costs: Docker setup, a second copy of
  the results kept honest by the sync rule, metric definitions in two
  languages kept equal by a test. About 5–6 h.
- **Plain-language explanations per forecast.** XGBoost's exact feature
  contributions (`pred_contribs`) summed into a few groups (recent level,
  day of week, holiday, SNAP, time of year), one sentence per forecast, a
  test that each explanation adds up to its forecast, and a stability
  check across adjacent origins. ETS and ARIMA explain themselves through
  their components and regression coefficients (FPP ch. 8 and ch. 10).
  Explanations describe the model, not the customer (FPP §7.8).

## Costs and the cache

Per method, per config, cached under `cache/predictions/` with a JSON
sidecar. Editing one model's module refits that model only. Changing
`fold_step`, `pool_by` or the item is a new config: every method refits for
it, once. Adding a config field at its default adopts old runs. Paths are
not in the key, so the cache is portable; this folder's `cache/` was copied
from the parent and holds the step-7 runs (52 and 104 folds).

| layout | all methods | XGBoost only | quantile XGBoost only |
|---|---|---|---|
| `Config(**DEV)`: 3 stores, 8 folds, step 7 | ~3 min | ~15 s | ~1 min |
| step 7, 10 stores, 52 folds | ~20 min | ~4 min | ~25 min |
| step 1, 10 stores, 358 folds | ~1 h 45 min | ~30 min | ~2.5 h |

Idle-machine figures. Two heavy jobs at once roughly quadruple them, so
run one at a time. Develop on `DEV`; run the full layout once per change
that survives it; report from the step-1 layout. Edits to the harness
(`step5_evaluate.py`) or the shared base invalidate every method's cache,
so batch them.

## Rules

- No employer, product, or third-party document is ever named in code,
  comments, notebooks, reports, commit messages, or this file. The two
  words to scan for are kept in the owner's private notes outside the
  repository; a case-insensitive `git grep` for each must return nothing
  before every commit.
- Commits carry the owner's identity only. No co-author or tool
  attribution lines of any kind.
- Report voice: third person, passive, technical; cite as (FPP §5.8); say
  what an orderer *can* do. Every number in a report is read back against
  the tables that produced it.
- The validator and the tests pass before any number is quoted.
- Correctness fixes to shared code flow back to the parent by cherry-pick
  (`git remote add parent https://github.com/wesGates/demand-forecasting.git`);
  nothing that changes a number in the parent's report is a fix.

## Local setup

- Folder: `~/Documents/projects/demand-forecasting-order`, own `.venv`
  (Python 3.14, `requirements.txt`), `data` is a symlink to the parent's
  data folder, `cache/` copied from the parent.
- Figures render to `figures/<notebook>/`; the report keeps its own copies.
