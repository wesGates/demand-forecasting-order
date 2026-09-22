# Demand forecasting on M5 — from point forecast to order quantity

This repository continues
[demand-forecasting](https://github.com/wesGates/demand-forecasting), which
established that XGBoost, ETS and seasonal ARIMA all beat six benchmarks on
a fast-moving item at ten stores, and that a point forecast is not an
order. The work here, in order:

1. Train XGBoost on the quantile objective at chosen service levels and
   compare it with ETS and ARIMA given the same treatment, scored with the
   quantile score and coverage.
2. Make every day a fold origin (`Config(fold_step=1)`) so the scoreboard
   averages over all order days; adopt it as the reported layout.
3. Run the pooled model across stores, the design that scales to thousands
   of items, and see whether it helps the quiet stores.
4. Repeat the study on an intermittent item.

The parent's README follows; it describes the pipeline this builds on.

---


Daily unit-sales forecasting for retail replenishment, built step by step
along the five-stage method in *Forecasting: Principles and Practice*
(Hyndman & Athanasopoulos, [otexts.com/fpppy](https://otexts.com/fpppy/)).

The question: on a fast-moving grocery item sold at ten stores, does a
gradient-boosted model beat the simple methods a person would use without one —
and does the answer depend on the store?

**Status:** steps 1–5 built and validated on one item across ten stores.
The write-up is [`report/report.md`](report/report.md); the dated numbers
behind it are in [`findings/`](findings/). Results below are the headline
tables from that report.

---

## What it does

- Loads the public M5 dataset (Walmart, 2011–2016), reshapes it to one row per
  store-item-day, and joins calendar, holiday, SNAP and price information.
- Screens each series for *availability* before anything else — a month of zero
  sales on a fast-moving item is a stocking gap, not demand, and it corrupts
  every statistic downstream. Treats the one closure day in the calendar
  (Christmas) as a missing observation rather than a zero (FPP §13.7).
- Measures which calendar events actually move the item, and gives the models
  holiday proximity — the run-up and the hangover — rather than an on/off flag.
- Classifies demand by how often and how consistently an item sells
  (ADI / CV², the Syntetos–Boylan scheme), computed on training data only.
- Forecasts seven days ahead from a single origin, daily, exactly as a
  replenishment run would: stand on Sunday, order for the week.
- Compares three learned models — **XGBoost**, **ETS** (Holt-Winters) and
  **seasonal ARIMA** with calendar regressors — against **six benchmarks**:
  mean, naïve, seasonal naïve (this day last week, the orderer's default
  screen), seasonal naïve at one year (this day last year, what an orderer
  checks before a holiday), drift, and a 28-day moving average.
- Scores with **RMSSE** (FPP §5.8) over 52 walk-forward folds (FPP §5.10), one
  full year, plus RMSE, MAE, bias, and win rates — every table split into
  normal and holiday weeks, so a pooled number can never hide where a win
  came from.
- Reports **win rate and per-week improvement quartiles** against the
  orderer's default screen, so a mean improvement cannot hide a quarter of
  weeks that got worse.
- Refuses to report a number until a validator has passed.

## Why the safeguards are the point

An earlier version of this project produced a 90% win rate that was wrong twice
over: lag features reached into the test window, and the only benchmark was
one that a dull model beats without skill. Everything here is built so those
two failures cannot recur:

- **Nothing sees the future.** A forecaster receives a `Context` with no target
  values in it, and training rows are filtered by *when their answer became
  observable*, not by when they were built. A date-comparison assertion refuses
  anything that reaches past the origin.
- **Six benchmarks, not one.** Seasonal naïve stakes everything on one past
  day; on a noisy series its error is about √2 worse than predicting the mean.
- **`python -m src.validate`** runs eight checks that each catch a different way
  of being *plausibly* wrong — closed-form benchmark answers, feature values
  recomputed by a second route, per-state SNAP flags against the raw calendar,
  holiday-proximity counts and closure flags recounted from the calendar, the
  quantile scoring on errors of known distribution, a synthetic noise floor no
  honest model can beat, a shuffled target no model should learn from, and
  byte-identical reruns.

## When a run is recomputed, and when it is not

A full run of every method costs about twenty minutes on the tiled layout
(52 origins, 7 days apart) and about seven times that when every day is an
origin (`Config(fold_step=1)`, the intended standard for reported numbers).
Each method's predictions are cached separately under `cache/predictions/`,
keyed on the config and on the code that method depends on:

- **Recomputed:** that method's module in `src/models/` changes (not its
  docstrings), the harness `step5_evaluate.py` or the shared `models/base.py`
  changes, the feature or loader version is bumped, or a config value
  changes.
- **Served from cache:** everything else - plots, notebooks, tests, the
  validator, prose, and edits to *other* methods' modules. Editing XGBoost
  refits XGBoost only.
- **Adopted:** a config that adds a new field at its default value reuses
  runs made before the field existed.

The cache is portable: file paths are not part of the key, so copying
`cache/` into a fork or another machine carries the runs across.

## Layout

```
src/
  step1_problem.py    Config — every decision in one frozen object, with FPP refs
  step2_data.py       load, reshape, join, pre-launch trim, parquet cache
  step3_explore.py    availability screen, ADI/CV² classification
  features.py         origin-based lags, rolling stats, calendar; leakage assertion
  step4_models.py     the registry: six benchmarks + XGBoost + ETS + ARIMA
  models/             one module per forecaster (cache invalidation is per module)
  step5_evaluate.py   walk-forward harness, RMSSE, normal/holiday split, tables
  order.py            weekly totals, calibrated quantile forecasts, pinball loss
  plots.py            every figure, on one palette
  validate.py         the gate
  render_figures.py   regenerate all figures to figures/<notebook>/
tests/                pytest suite: benchmarks, features, harness, models, ordering
notebooks/
  01_explore.py       FPP step 3 — graph the data before modelling anything
  02_evaluate.py      FPP step 5 — run the comparison, read the diagnostics
  03_order.py         FPP step 5, continued — from forecast to order quantity
findings/             what the notebooks found for a particular item, dated
report/               the write-up, with its own copies of the figures
```

Notebooks are plain `.py` files with `# %%` cell markers: open in VS Code and
run cells with Shift+Enter in the Interactive Window. A cell near the top pops
figures out into their own windows.

## Setup

```
python -m venv .venv                  # Python 3.14
.venv\Scripts\activate
pip install -r requirements.txt
```

Data is the Kaggle **M5 Forecasting – Accuracy** competition set. Place these
three files in `data/` (gitignored):

```
sales_train_evaluation.csv
calendar.csv
sell_prices.csv
```

Then:

```
python -m src.validate          # must pass before any number is quoted
python -m pytest tests          # 82 tests pinning documented behaviour
python -m src.render_figures    # all figures -> figures/<notebook>/
python -m src.step5_evaluate    # the comparison, as tables
python -m src.order             # prototype: forecast -> order quantity (follow-on work)
```

## Results

One item (`FOODS_3_586`), ten stores, 52 walk-forward folds of 7 days — one
full year, 25 May 2015 to 22 May 2016 — RMSSE scaled by the seasonal-naïve
error on training data. "Holiday" folds are the ten weeks per store that
touch the window around a major event; "normal" folds are the other 42.

| method | all weeks | normal weeks | holiday weeks | bias (units/day) |
|---|---|---|---|---|
| ARIMA (seasonal, calendar regressors) | 0.65 | 0.63 | 0.74 | −0.1 |
| XGBoost | 0.67 | 0.65 | 0.72 | +0.9 |
| ETS (Holt-Winters) | 0.67 | 0.64 | 0.80 | −0.1 |
| moving average (28) | 0.80 | 0.78 | 0.87 | −0.1 |
| seasonal naïve (last week) | 0.86 | 0.82 | 1.02 | −0.1 |
| mean | 0.96 | 0.96 | 0.99 | +3.3 |
| seasonal naïve (last year) | 1.04 | 1.03 | 1.08 | −0.1 |
| naïve / drift | 1.10 | 1.10 | 1.10 | +9.5 |

What the year says that the spring slice could not:

- The three models are within 0.03 of each other over the year and all beat
  every benchmark at every store. Which one is *best* depends on the store:
  XGBoost leads at all three Texas stores and two of the three Wisconsin
  ones; ETS or ARIMA leads at every California store.
- The split shows where each earns its keep. On ordinary weeks the classical
  models edge XGBoost. On holiday weeks XGBoost is best and ETS falls
  furthest — ETS is the one model that cannot be told a holiday is coming,
  and the gap between it and ARIMA on holiday weeks (0.80 vs 0.74) is the
  price of that.
- "This day last year" — what an orderer checks before a holiday — is a
  *worse* benchmark than "this day last week" on this item, even in holiday
  weeks. The item's level has drifted year over year, and the annual lookup
  carries the old level with it.
- Naïve and drift over-forecast by nine units a day because every origin is a
  Sunday, the busiest day, and they repeat it for the week.
- XGBoost's over-forecast is smaller over the year (+0.9/day) than in the
  spring slice (+1.2), but it remains the one model with a positive bias.

### Win rate and improvement over the default screen

| model | win rate vs seasonal naïve | mean | median | Q1 | Q3 |
|---|---|---|---|---|---|
| ARIMA | 82% | 20% | 24% | 10% | 37% |
| ETS | 83% | 19% | 21% | 8% | 33% |
| XGBoost | 75% | 16% | 23% | 0% | 40% |

XGBoost's median improvement equals ARIMA's, but its lower quartile is zero:
in a quarter of weeks it does no better than the default screen. It also
over-forecasts by about 4% at the busiest store, which the bias column shows
and the error score does not.

A point forecast is not an order. A prototype of the next step, turning the
forecast into a quantile order at a service level and scoring it with the
quantile score, is in `src/order.py` and notebook `03_order`; it is the
subject of the follow-on work and is not part of this report.

## What is deliberately not claimed

This is a learning project on public data. M5 has no inventory positions,
receipts or transfers, so the replenishment calculation a forecast feeds into
can only be simulated. Price is excluded as a feature for this item because it
changes on two fixed dates in five years — a clock, not a variable — and that
would not hold for another product. Pooling across stores is wired in but not
yet run.
