# Glossary

Terms the findings, the plan and the validator use without explaining.
Each entry says what the thing is, why it exists, and where it lives.

## Origin, fold, layout

**Origin.** The day a forecast is made from. The forecaster sees every day
up to and including the origin and forecasts the next seven. A store
placing an order on Sunday evening stands at a Sunday origin.

**Fold.** One origin plus its seven target days, for one series. Scoring a
fold means comparing the seven forecasts with what sold.

**Weekly layout (`fold_step=7`).** Origins seven days apart, 52 folds in
the scored year, no overlap. Fast, and the first study's layout. Every
origin falls on the same weekday, which turned out to matter (item 2).

**Every-day layout (`fold_step=1`).** Every day of the scored year is an
origin, 358 of them, so folds overlap and the weekday rotates. The
reported layout since item 2. About seven times the fitting.

**Suite.** A named layout a change is scored on: `dev`, `weekly`,
`everyday` in `src/suites.py`. The item, the pooling and the method are
chosen per run. Two runs on the same suite are comparable.

## The cache

**What it is.** Every forecast a method makes is written once to
`cache/predictions/<method>_<digest>.parquet`, one row per series, origin,
method and day ahead, with the actual, the RMSSE scale and a fallback
flag beside it. A JSON file with the same name records the config and the
code digest the run was made under.

**The key.** The digest in the file name hashes the config (every `Config`
field except file paths), the method's name, and the code the method
depends on (the harness, the shared base, the method's own module,
`features.py`, `step2_data.py`, `step1_problem.py`, and the feature and
loader version numbers). Change any of those and the key changes; the old
file stays and a new one is written on the next run. Code is hashed as a
syntax tree with docstrings removed, so a comment edit does not invalidate
a run and a logic edit does. (`_method_cache_path`, `_method_code` in
`src/step5_evaluate.py`.)

**Cached runs are checked against the data.** The key names the config and
the code. On every read the harness compares the cached actuals with the
panel's sales, and any difference is an error. Without that, a changed
data file would get old forecasts served back.

**"Cache equals code."** The claim that a cached file holds what the
current code would produce. The key guarantees it when the key matches;
a from-scratch refit tests it directly (`tools/recompute_check.py`).

**Refit.** Running a method with `use_cache=False` so every fold is fitted
again. `tools/refit_all.py` refits every reported configuration and
registers each run under a note; `tools/compare_forecasts.py` then says,
forecast by forecast, what changed.

## The validator's negative controls

`python -m src.validate` runs ten checks. Three of them run on synthetic
data where the right answer is known, so a model that does too well must
be cheating.

**Synthetic panel.** Six made-up series, 1,000 days each: a level, a
weekly shape, a slow trend, and Gaussian noise with a chosen standard
deviation (2.0). Everything except the noise is a fixed function of the
date. (`_synthetic_panel` in `src/validate.py`.)

**Noise floor.** The only unpredictable part of the synthetic series is the
noise, so the best possible RMSE is the noise's own standard deviation.
The check runs the walk-forward on the panel and divides the observed RMSE
by that floor. A ratio below 0.90 means some feature saw the future. The
ratios on 2026-09-23 were 1.17 for every-day origins and 1.12 for the
pooled model.

**Shuffled target.** The same panel with the sales scrambled, so there is
no pattern to learn. A model that beats a mean-predicting benchmark on
noise is learning from something it should not see. The benchmark has to
be mean-based; the seasonal naïve is about 1.4 times worse than the mean
on pure noise, so against it a healthy pipeline would look like a leak.

**Layouts and pooling.** Both controls repeated with every day an origin
and with the model pooled across series, the two settings the default
checks do not exercise. (`check_layouts_and_pooling`.)

**Harness parity.** The real harness run in-process and with three
workers on the synthetic panel, per series and pooled. The forecasts must
be identical.

## Pooling

**Per-store model.** One XGBoost per series (store and item), trained on
that series' rows only.

**Pooled model (`pool_by="item_id"`).** One XGBoost per origin for the
whole item, trained on all ten stores' rows with store identity as a
feature. It learns the weekly and holiday shape from ten stores' worth of
history. Item 3.

**Pool membership.** The set of series whose rows the pooled model trains
on. Part of the memo key, so a model trained on a different set of stores
is a different model.

**One fit per origin.** A pooled model is the same model for every store at
a given origin, so it is fitted once and reused ten times. The first
pooled run fitted it ten times per origin because the harness asks for
forecasts store by store; a memo keyed on (pooling field, origin, pool
membership) fixed that. (`_pooled_models` in `src/models/xgboost_model.py`.)

**No leakage across stores.** Pooling adds other stores' rows to the
training set. Each row carries the date its target was observed, and the
training set keeps only rows whose target date is at or before the origin,
with an assertion right after (`design()` in `src/models/xgboost_model.py`).
The pooling control tests the same thing on synthetic data.

## The level-relative target

**Level.** The recent average of a series. Here it is the mean of the 28
days before the origin (`roll_mean_28`, a feature every row carries,
computed at the row's own origin).

**Plain target.** The trees predict sales directly. A tree predicts from
leaves grown on the values it saw in training, so it cannot predict a
value outside that range. On an item whose sales fall over time the
forecast stays where the history was. Item 4 measured a 60% over-forecast
on FOODS_1_021.

**Level-relative target (`xgboost_rel`).** The trees predict sales minus
the level at the origin, and the forecast is that prediction plus the
level, clipped at zero. The last four weeks carry the level and the trees
learn the weekly, holiday and calendar shape around it. Exponential
smoothing tracks a level by construction (FPP §8.1), which is why ETS never
had this problem. Item 5.

## Scores

**RMSSE.** Root mean squared error of a fold divided by the root mean
squared error the seasonal naïve made on the series' training data (FPP
§5.8). 1.0 means no better than repeating last week's same weekday; 0.8
means 20% better. The scale belongs to the series, so a 100-unit store and
a 15-unit store compare.

**Bias.** Mean of forecast minus actual, in units a day. Positive means
over-forecasting. Reported beside RMSSE because a symmetric error score
cannot show a forecast that runs high every day.

**Win rate and improvement quartiles.** For each fold, the percentage by
which a method's RMSSE is below a benchmark's on the same series and
window. The win rate is the share of folds where it is below at all; the
quartiles describe the spread. The lower quartile says how much a method
improves in its worst quarter of folds.

**Normal and holiday weeks.** Every fold is labelled by whether a major
calendar event falls in its window (two days before to one day after),
and every table is reported for both kinds.

**Unscored fold.** A fold whose forecast is missing or whose RMSSE scale
is zero or NaN. Counted in `n_unscored` and left out of every average and
every paired comparison. The old code scored these as infinity, which made
a method's mean RMSSE infinite after one such fold.

**Fallback.** A forecaster that cannot fit a fold (a failed ETS or ARIMA
estimation) returns the 28-day mean and says so through
`base.note_fallback`. The harness records `fallback` on every row of that
forecast and the tables count them (`n_fallback`). A method that won on
fallbacks is visible as such. No fallback occurred in the item 8 refit.

## The run registry

**Run registry.** `cache/registry.sqlite`, one row per cached run in `run`
and one row per store-origin in `fold_score`, built from the cache and git
by `src/registry.py`. It answers which runs exist, at what commit, and how
they scored, without reading the cache. Rebuilt with
`python -m src.registry backfill`; never edited by hand. A run is recorded
once; recording it again appends the note and keeps the first row's time
and commit.

**Predecessor.** The most recent earlier run of the same method and the
same full config. `python -m src.registry last <run_id>` compares a run
with it.

**Paired comparison.** Two runs compared on the store-origins they share:
for each, the percentage by which B's RMSSE is below A's. Reported as B's
win rate and the quartiles of that percentage, with the count of pairs
where A's RMSSE was zero and the percentage is undefined. A change inside
the fold-to-fold noise shows a win rate near 50% and a median near zero.

**`code_current`.** Whether a run's code digest is what the code computes
now for that method. A snapshot; `python -m src.registry refresh`
recomputes it after a code change.

## The parallel harness

**Task.** The unit of work handed to a worker process: one fold of one
series when models fit per series, or one fold of every series when they
pool, because a pooled model is fitted once per origin and shared. Each
task rebuilds the same `Context` the serial loop built, runs every
requested method on it, and returns the rows. (`_forecast_task` in
`src/step5_evaluate.py`.)

**One thread per fit.** XGBoost runs with `n_jobs=1` and every numerical
library is pinned to one thread inside each worker. Speed comes from
running many single-threaded fits at once. On this machine that is about
8× faster for XGBoost and 6× for ARIMA than the old all-cores-per-fit
setting, because these fits are too small to use twenty threads well. It
also makes the forecasts independent of the machine's core count.

**`n_jobs`.** How many worker processes the harness uses (default: every
core). It changes wall time only, never a forecast, so it is not part of
the cache key. `n_jobs=1` runs the same task function in-process.

**ARIMA order selection.** ARIMA chooses its order once per series on that
series' first scored window and holds it for the year. The harness makes
that choice first, in parallel over series, and hands the orders to every
worker. A failed choice is an error.

**Reproducibility caveat.** With the seed and the thread count fixed, a
fit is deterministic on the same platform. XGBoost's column subsampling
can give different, individually reproducible results on a different
operating system, so identity across machines is claimed only for the
same OS and library versions.

## Rules added by the 2026-09-23 review

**Closure imputation is backward-only.** A closure day (Christmas) is
filled with the mean of the same weekday over the four preceding weeks,
skipping days inside a holiday window. Christmas 2015 sits inside the
scored year, and a value borrowed from January 2016 would have entered the
lags, the training rows and the seasonal naïve of every fold with an
origin in the next four weeks (FPP §5.10). The book's own guidance (§13.7)
is to set a closed day to zero and give an ARIMA dummies for the day and
the day after, or to interpolate an outlier from both sides; both are data
preparation before a single fit. The backward rule is that estimate
restricted to the past. Two-sided filling would have been fine for the
four Christmases before the scored year; one rule was kept for simplicity,
and the refit put the whole choice in the third decimal. `CACHE_VERSION` 3.

**Events are chosen before the cutoff.** `event_effects(df, calendar,
cutoff)` measures each event on the data before the first scored day, so
the holiday feature set is not selected on the test period (FPP §5.8). On
the study's item the same events clear the threshold either way.

**The daily grid is asserted.** Every series must be an unbroken run of
days, because lags, benchmarks and the RMSSE scale are positional. A source
with gaps must be reindexed and flagged first (FPP §13.7).
