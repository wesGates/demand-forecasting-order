# Demand forecasting on M5 — a forecaster's-toolbox build

Daily unit-sales forecasting for retail replenishment, built step by step
along the five-stage method in *Forecasting: Principles and Practice*
(Hyndman & Athanasopoulos, [otexts.com/fpppy](https://otexts.com/fpppy/)).

The question: on a fast-moving grocery item sold at ten stores, does a
gradient-boosted model beat the simple methods a person would use without one —
and does the answer depend on the store?

**Status:** steps 1–5 built and validated on one item across ten stores.
Write-up in progress. Results below are preliminary.

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
- Turns the forecast into an **order quantity**: the weekly total, at a
  chosen service level, from each method's own calibrated error quantiles
  (FPP §5.5), scored with the quantile score (§5.9) across a range of service
  levels — because a symmetric error cannot see an asymmetric cost.
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

## Layout

```
src/
  step1_problem.py    Config — every decision in one frozen object, with FPP refs
  step2_data.py       load, reshape, join, pre-launch trim, parquet cache
  step3_explore.py    availability screen, ADI/CV² classification
  features.py         origin-based lags, rolling stats, calendar; leakage assertion
  step4_models.py     six benchmarks + XGBoost + ETS + ARIMA behind one interface
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
python -m src.order             # from forecast to order quantity (two-year run)
```

## Preliminary result

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

### From forecast to order

A point forecast is not an order. Ordering the mean stocks out about half
the weeks — measured, not assumed: the classical models under-forecast the
weekly total in 50% of weeks, XGBoost in 44%. The order is a quantile of the
week's demand at a chosen service level τ, and τ is an economic choice (the
understock cost over the sum of both costs), unknown for an anonymised item,
so it is reported as a range. Each method's quantiles come from its own
weekly errors over a *calibration year* (May 2014 – May 2015), applied to
the scored year — never estimated on the weeks they are judged on.

Relative pinball loss (FPP §5.9's quantile score over mean weekly sales;
lower is better; compare within a column, never along a row):

| method | τ = 0.3 | τ = 0.5 | τ = 0.7 | τ = 0.9 | coverage at 0.9 |
|---|---|---|---|---|---|
| ARIMA | **0.094** | **0.106** | **0.096** | 0.053 | 0.92 |
| XGBoost | 0.114 | 0.116 | 0.104 | **0.052** | 0.92 |
| moving average (28) | 0.105 | 0.117 | 0.107 | 0.062 | 0.90 |
| ETS | 0.104 | 0.120 | 0.107 | 0.059 | 0.89 |
| seasonal naïve (last week) | 0.109 | 0.123 | 0.110 | 0.061 | 0.89 |

- **The ranking does change with the cost asymmetry.** At τ = 0.9, an
  ambient item's service level, XGBoost and ARIMA tie. At τ = 0.3, a
  perishable's, XGBoost falls to fifth, behind two benchmarks. RMSSE said
  nothing about this, and could not.
- ARIMA is the most robust method across the whole range, and the best on
  holiday weeks at every τ.
- Coverage is close to target for every model: a 0.9 order covered 89–92% of
  weeks. One year's errors described the next well enough to order from.
- XGBoost's calibration transfers less well than ARIMA's: its errors shrink
  as it gets more training data, so last year's spread overstates this
  year's. Letting the calibration window grow through the scored year, as a
  live system would, closes most of the gap (0.099 at τ = 0.3).

## What is deliberately not claimed

This is a learning project on public data. M5 has no inventory positions,
receipts or transfers, so the replenishment calculation a forecast feeds into
can only be simulated. Price is excluded as a feature for this item because it
changes on two fixed dates in five years — a clock, not a variable — and that
would not hold for another product. Pooling across stores is wired in but not
yet run.
