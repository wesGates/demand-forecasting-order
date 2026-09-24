# Glossary

Terms that the findings files, `PLAN.md` and the validator output use
without stopping to explain. Each entry says what the thing is, why it is
there, and where in the code it lives.

## Origin, fold, layout

**Origin.** The day a forecast is made from. The forecaster sees every day
up to and including the origin and forecasts the next seven. A store
placing an order on Sunday evening is standing at a Sunday origin.

**Fold.** One origin plus its seven target days, for one series. Scoring a
fold means comparing the seven forecasts with what sold.

**Weekly layout ("tiled", `fold_step=7`).** Origins seven days apart, so the
52 folds of the scored year sit end to end without overlap. Fast, and the
layout the first study used. Its flaw: every origin falls on the same
weekday.

**Every-day layout (`fold_step=1`).** Every day of the scored year is an
origin (358 of them), so folds overlap and the origin weekday rotates. The
reported layout since item 2. Cost: about seven times the fitting.

## The cache

**What it is.** Every forecast a method produces is written once to
`cache/predictions/<method>_<digest>.parquet`, one row per series, origin,
method and day ahead, with the actual and the RMSSE scale beside it. A JSON
file with the same name records the configuration and the code digest the
run was made under.

**The key.** The digest in the file name is a hash of three things: the
configuration (every `Config` field except file paths), the method's name,
and a digest of the code the method depends on (the harness, the shared
base, the method's own module, `features.py`, `step2_data.py`, and the
feature and loader version numbers). Change any of those and the key
changes; the old file stays, a new one is written on the next run. Code is
hashed as its syntax tree with docstrings removed, so editing a comment does
not invalidate a run and editing logic does. (`_method_cache_path` and
`_method_code` in `src/step5_evaluate.py`.)

**Adoption.** When a configuration field is added with a default value, an
older run that never knew the field is still valid; the loader adopts it
rather than refitting. (`_adopt_cached`.)

**"Cache equals code."** The claim that a cached file contains exactly what
the current code would produce if run again. The key guarantees it when the
key matches; the from-scratch refit (below) tests it directly.

**From-scratch refit.** Running a method with `use_cache=False`, so every
fold is fitted again, and comparing the result to the cached file row by
row. Done 2026-09-23 for the per-store and pooled XGBoost on the weekly
layout: 3,640 forecasts each, identical to the cache.

**Re-keying.** After a change that provably alters no forecast (a table
moved, scoring code split into its own module), cached files can be copied
to the key the new code computes instead of refitting. Allowed only for such
changes; a script for it is kept outside the repository.

## The validator's negative controls

The validator (`python -m src.validate`) runs nine checks. Two of them are
*negative controls*: tests on synthetic data where the right answer is known
and any model that does "too well" must be cheating.

**Synthetic panel.** Six made-up series, 1,000 days each: a level, a weekly
shape, a slow trend, and Gaussian noise with a chosen standard deviation
(`noise_sd`, 2.0). Everything except the noise is a deterministic function
of the date. (`_synthetic_panel` in `src/validate.py`.)

**Noise floor.** Because the only unpredictable part of the synthetic series
is the noise, the best possible RMSE is the noise's own standard deviation.
The check runs the real walk-forward on the panel and divides the observed
RMSE by that floor. A ratio meaningfully below 1 (the threshold is 0.90)
means the model saw the future: some feature is reaching past the origin.
A ratio at or above 1 is a pass. The ratios reported on 2026-09-23 were
1.17 for every-day origins and 1.12 for the pooled model.

**Shuffled target.** The same panel with the sales values scrambled, so
there is no pattern left to learn. The check compares the model's RMSE with
a mean-predicting benchmark's. A model that beats the benchmark on noise is
learning from something it should not have. Ratios of 1.00 (pass) were
reported for both the every-day and the pooled setting. The benchmark must
be mean-based: against the seasonal naïve this test fires on a healthy
pipeline, because the seasonal naïve is about 1.4 times worse than the mean
on pure noise.

**Layouts and pooling.** Both controls repeated under the two settings the
default checks do not exercise: every day an origin (overlapping windows)
and one model pooled across series. A leak that only opens when windows
overlap, or when other series' rows join the training set, is caught here.
(`check_layouts_and_pooling`.)

## Pooling

**Per-store model.** One XGBoost per series (store and item), trained on
that series' rows only. The first study's design.

**Pooled model (`pool_by="item_id"`).** One XGBoost per origin for the whole
item, trained on all ten stores' rows at once, with store identity as a
feature. It learns the weekly and holiday shape from ten stores' worth of
history and applies it to each. Item 3.

**Pool membership.** The set of series whose rows the pooled model trains
on: here, the ten store series of one item. It is part of the memo key
(below), so a model trained on a different set of stores is a different
model.

**One fit per origin.** A pooled model is the same model for every store at
a given origin, so it should be fitted once per origin and reused ten
times. The first pooled run fitted it ten times per origin, because the
harness asks for forecasts store by store. The fix is a memo keyed on
(pooling field, origin, pool membership, parameter overrides): the first
store at an origin fits the model; the other nine reuse it.
(`_pooled_models` in `src/models/xgboost_model.py`; `reset_run_state()`
clears it between runs.) Run time fell from ten fits per origin to one, 16
minutes for 358 origins.

**No leakage across stores.** Pooling adds other stores' rows to the
training set. Each row carries the date its target was observed, and the
training set is filtered to rows whose target date is at or before the
origin, with an assertion right after (`design()` in
`src/models/xgboost_model.py`). The validator's pooling control tests the
same thing empirically.

## The level-relative target

**Level.** The recent average sales of a series: here, the mean of the 28
days before the origin (`roll_mean_28`, a feature every row already
carries, computed at the row's own origin).

**Plain target.** The trees are trained to predict sales directly. A tree
model predicts from leaves grown on the values it saw in training, so it
cannot predict a value outside that range. On an item whose sales fall
over time, the forecast stays where the history was. Item 4 measured this:
60% over-forecast on a declining item.

**Level-relative target (`xgboost_rel`).** The trees are trained to predict
*sales minus the level at the origin*, and the forecast is that prediction
plus the level, clipped at zero. The level is carried by the last four
weeks of data, which move with the item; the trees learn only the weekly,
holiday and calendar shape around it. Every feature and setting is
otherwise the same as the plain model. (`src/models/xgboost_relative.py`.)
Item 5: on the declining item the pooled model's RMSSE fell from 0.570 to
0.500, level with the best simple methods; on the fast mover from 0.621 to
0.612, the best of any method.

## Scores

**RMSSE.** Root mean squared error of a fold divided by the root mean
squared error the seasonal naïve forecast made on the series' training
data (FPP §5.8). 1.0 means no better than repeating last week's same
weekday; 0.8 means 20% better. The denominator belongs to the series, so a
100-unit store and a 15-unit store are comparable.

**Bias.** Mean of forecast minus actual, in units per day. Positive means
over-forecasting. Reported beside RMSSE because a symmetric error score
cannot show a forecast that runs high every day.

**Win rate and improvement quartiles.** For each fold, the percentage by
which a method's RMSSE is below a benchmark's on the same series and
window. The win rate is the share of folds where it is below at all; the
quartiles describe the spread. The lower quartile says how much a method
improves in its worst quarter of folds, which a mean hides.

**Normal and holiday weeks.** Every fold is labelled by whether a major
calendar event falls within its window (two days before to one day after),
and every table is reported for both kinds, because a pooled number cannot
say where a method's advantage came from.

## Provenance

**Provenance block.** The header of every findings file since item 3:
branch, commit, the configuration line that rebuilds the run, and the cache
file each method's forecasts were read from. Printed by
`step5_evaluate.provenance()` at run time, so the branch and commit are
those of the run, not of the write-up. Anyone can check out the commit,
rebuild the configuration, and either read the same cache file or refit and
compare.

**DEV preset.** `Config(**DEV)`: three stores, eight folds, weekly layout.
About twenty seconds for XGBoost, for iterating on a model change before
paying for a full run.

## The run registry

**Run registry.** `cache/registry.sqlite`, one row per cached run in `run`
and one row per store-origin in `fold_score`, built from the cache and git
by `src/registry.py`. It answers "which runs exist, at what commit, and how
did they score" without reading the cache. Rebuilt with `python -m
src.registry backfill`; never edited by hand.

**Suite.** A named, fixed layout a change is scored on (`dev`, `weekly`,
`everyday` in `src/suites.py`). The item, the pooling and the method are
chosen per run; the suite pins folds and stores so that two runs are
comparable.

**Predecessor.** The most recent earlier run of the same method on the same
suite, item, stores and pooling. `python -m src.registry last <run_id>`
compares a run with it.

**Paired comparison.** Two runs compared on the store-origins they share:
for each, the percentage by which B's RMSSE is below A's. Reported as B's
win rate and the quartiles of that percentage. A change inside the
fold-to-fold noise shows a win rate near 50% and a median near zero.

## The parallel harness (item 7)

**Task.** The unit of work handed to a worker process: one fold of one
series when models fit per series; one fold of *every* series when they
pool (a pooled model is fitted once per origin and shared by the stores, so
the stores sit in the same process). Each task rebuilds the same `Context`
the serial loop built, runs every requested method on it, and returns the
rows. (`_forecast_task` in `src/step5_evaluate.py`.)

**One thread per fit.** XGBoost runs with `n_jobs=1` and every numerical
library is pinned to one thread inside each worker (`threadpoolctl`).
Speed comes from running many single-threaded fits at once, not from
threads inside a fit: on this machine that is about 8× faster for XGBoost
and 6× for ARIMA than the old all-cores-per-fit setting, because these fits
are too small to use twenty threads well. It also makes the forecasts
independent of how many cores the machine has, which the old setting was
not (XGBoost's floating-point sums depend on the thread count).

**`n_jobs`.** How many worker processes the harness uses (default: every
core). It changes wall time only, never a forecast, so it is not part of
the cache key. `n_jobs=1` runs the same task function in-process, which is
what the tests compare against.

**ARIMA order selection.** ARIMA chooses its order once per series on that
series' first scored window and holds it for the year. With folds spread
over processes, the harness makes that choice first, in parallel over
series, and hands the orders to every worker; otherwise each worker would
choose its own on whatever window it saw first.

**Reproducibility caveat.** With the seed and the thread count fixed, a
fit is deterministic on the same platform. XGBoost's column subsampling
can still give different (but each reproducible) results on a different
operating system, so cross-machine identity is claimed only for the same
OS and library versions.
