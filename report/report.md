# XGBoost Against Classical Methods for Daily Store Demand

Wesley Gates. Draft, 22 September 2026.

Public data: M5 (Walmart daily unit sales, 2011–2016). Code, tests and
figures are in this repository.

---

## 1. Executive Summary

A forecasting pipeline was built for daily unit sales of one grocery item at
ten stores, and used to test whether a gradient-boosted model (XGBoost) beats
the forecast an orderer can produce without one. The comparison was scored
on a full year of weekly forecasts, 52 walk-forward folds from 25 May 2015
to 22 May 2016, against six benchmarks and two classical models. The
reference benchmark is the seasonal naïve forecast, this day last week,
because that is what an orderer's default screen can show.

Table 1: Headline results against the seasonal naïve benchmark, 520
store-weeks. Improvement is the per-week reduction in scaled error (RMSSE).

| model | win rate | mean improvement | median | Q1 | Q3 |
|---|---|---|---|---|---|
| ARIMA (seasonal, holiday regressors) | 82% | 20% | 24% | 10% | 37% |
| ETS (Holt-Winters) | 83% | 19% | 21% | 8% | 33% |
| XGBoost | 75% | 16% | 23% | 0% | 40% |

All three models beat the benchmark at every store. Over the year their
mean scaled errors are 0.65, 0.67 and 0.67, close enough that the ordinary
weeks are a tie. XGBoost has the widest spread: its best quarter of weeks
improves the most, and its worst quarter does not improve at all. It wins at
the Texas stores and on holiday weeks. The classical models win at the
California stores and on ordinary weeks. XGBoost also carries a systematic
over-forecast at the busy stores that the error score does not show and the
bias column does.

Everything quoted here passed a seven-check validator and the test suite
before it was written down.

## 2. Background and Problem

Replenishment consumes a forecast one day at a time, so the forecast is
daily, seven days ahead from a single origin: stand on Sunday, forecast
Monday through Sunday. Two things about it matter more than in a general
forecasting exercise. Bias matters in its own right, because a forecast
that runs low every day turns into stockouts. And the comparison has to be
against what a person can do, because an orderer can work from a screen
that shows the same day last week and, before a holiday, the same day last
year.

The M5 dataset was used because it is public and structurally close to the
problem: daily sales per store and item, a calendar of events, SNAP benefit
days by state, and weekly shelf prices. The item, `FOODS_3_586`, was chosen
because it is continuously stocked at all ten stores and sells 14 to 103
units a day depending on the store, which gives a volume gradient to test
across.

Requirements set at the start:

| requirement | how it is enforced |
|---|---|
| No feature may use data from after the forecast origin | a date assertion on every fold; two negative-control checks in the validator |
| The comparison must include what an orderer can do by default | seasonal naïve at lag 7 and at lag 364 are benchmarks |
| The comparison must include a benchmark a dull model cannot beat by accident | 28-day moving average and long-run mean are benchmarks |
| Methods must be routed by the item's demand type | ADI / CV² classification, computed on training data only |
| Every reported number must be reproducible | fixed seed, byte-identical rerun check, cached runs keyed on the code |

An earlier version of this project reported a 90% win rate that turned out
to be wrong for two reasons: lag features reached into the test window, and
the only benchmark was one a dull model beats without skill. The current
design is a response to that.

## 3. Data Preparation

The three raw files are reshaped to one row per store, item and day and
joined to the calendar, with each row taking the SNAP flag for its own
state. Two decisions were made in the loader rather than downstream.

Christmas is a closure, not a zero. Every store records zero on Christmas
Day because the stores were shut, and left as a zero it pulls down every
rolling mean for a week and feeds the seasonal naïve forecast for the
following week. The day is treated as a missing observation (FPP §13.7):
imputed with the same-weekday mean of the surrounding weeks, flagged, and
excluded from every score.

Which holidays matter was measured. Each of the calendar's 30 events was
scored against a same-weekday baseline, on the day and on the two days
before it (Figure 1). Seven clear a 15% threshold: Thanksgiving, Christmas,
Labor Day, Independence Day, Valentine's Day, Easter and New Year. The
run-up before Christmas (1.7 times baseline two days before) is larger than
most holidays' own day, so the models get three features from these events:
a flag, days until the next, and days since the last. An on/off flag alone
cannot represent a run-up.

![Figure 1](figures/01_explore_6_event_effects.png)

Figure 1: Each calendar event's sales relative to a same-weekday baseline,
on the day (filled) and the larger of the two days before (open). The seven
selected events are in colour. Christmas has no day marker because the
stores were shut.

## 4. Exploration

Availability was checked first, since a long run of zero sales on a fast
mover is a stocking gap rather than demand. The longest zero run at any
store is 10 days; the screen warns at 30 and did not fire.

The demand was then classified. Two numbers decide whether a series is a
candidate for a learned model at all: the average demand interval (days per
sale) and the squared coefficient of variation of the sale sizes. Every
store of this item is smooth (Figure 2), with ADI at 1.00 and CV² between
0.06 and 0.24 against a cut of 0.49. So the study cannot ask whether the
best method changes with demand class; it asks which method wins on a fast
mover and whether that depends on the store. The classification is the
reusable part: an intermittent item would be routed to different methods and
a different metric, which is where a learned model is expected to lose.

![Figure 2](figures/01_explore_2_classmap.png)

Figure 2: Demand classification. Every store sits in the smooth corner; the
spread within the corner is the volume gradient the comparison runs along.

The dominant structure is weekly. Weekend days sell about 50% more than
midweek at every store, and the shape is shared across stores once each is
indexed to its own mean. Lag 1 is strong on its own, so the recent level
matters independently of the weekly pattern, and lags 6 and 8 are nearly as
tall as 7, so the weekend peak is broad. There is a multi-year decline
underneath: the busiest store averaged 109 units a day in 2011 and 82 in
early 2016. SNAP benefit days run 6% above ordinary days and are positive at
nine of ten stores, so the flag was kept. Price took three values in five
years and never changed inside a forecast window, so it was left out; a
check with it on made XGBoost slightly worse (0.674 against 0.665).

## 5. Models and Benchmarks

Six benchmarks were used (Table 2). The seasonal naïve at lag 7 is the
reference for the headline claims: it is the default screen, and it is also
the RMSSE denominator, so improvement over it and RMSSE are the same
quantity seen two ways. The 28-day moving average is the hardest of the six
to beat and is reported as the stress check.

Table 2: Benchmarks.

| benchmark | what it represents |
|---|---|
| seasonal naïve, lag 7 | the default screen: this day last week (reference) |
| seasonal naïve, lag 364 | the holiday habit: this day last year |
| 28-day moving average | the recent level, ignoring the weekday (stress check) |
| mean | the whole history's level |
| naïve, drift | the last value, flat and with a trend |

XGBoost was trained per store with early stopping on the last 90 days of
each training window. Its features are:

- lags of daily sales at 1, 2, 3, 7, 14, 21 and 28 days before the origin;
- rolling mean and standard deviation over the last 7, 28 and 56 days, and
  the ratio of the 7-day to the 56-day mean;
- same-weekday means over the last 4 and 8 weeks;
- calendar: weekday, weekend flag, day of month, month, day of year;
- SNAP flag, any-event flag, and the three holiday-proximity features;
- the horizon, 1 to 7, since the features above are shared by the seven
  target days of a fold.

Two classical models were fitted for comparison. Holt-Winters exponential
smoothing (ETS) with weekly seasonality is the usual first choice for a
series like this; the implementation cannot take regressors, so it cannot
be told a holiday is coming. Seasonal ARIMA was added with three
regressors, the holiday flag, a two-day run-up flag and SNAP, so that a
classical model gets the same calendar information XGBoost gets. The ARIMA
order is chosen once per store by AICc on its first training window and
held for the year. A plain ARMA was rejected: without the seasonal term it
would need very high orders to imitate a weekly cycle on daily data.

Every feature for a target day is computed from data available at the
origin, and a date comparison on every fold refuses anything that reaches
past it. Two validator checks cover leaks a date comparison would miss: a
synthetic series with a known noise floor that no honest model can beat,
and a shuffled target that no model should be able to learn.

## 6. Evaluation

Each method is run through 52 walk-forward folds of seven days. Each fold
trains on all history before its Sunday origin and is scored on the
following Monday to Sunday (rolling-origin evaluation, FPP §5.10). One year
was chosen so that every season is scored once, holidays included. The
first version of this study scored eight spring weeks and never saw a
holiday.

The score is RMSSE (FPP §5.8): the forecast's root mean squared error
divided by what a seasonal naïve forecast would have made on the training
data. A value of 1.0 means no better than repeating last week's same
weekday; 0.8 means 20% better. The denominator belongs to the series, so a
100-unit store and a 15-unit store are comparable. Every table is reported
on ordinary weeks and on holiday weeks separately, because a pooled number
cannot say where a method's advantage came from.

Table 3: Mean RMSSE over 520 store-weeks. Bias is forecast minus actual in
units per day.

| method | all weeks | normal | holiday | bias |
|---|---|---|---|---|
| ARIMA | 0.65 | 0.63 | 0.74 | −0.1 |
| XGBoost | 0.67 | 0.65 | 0.72 | +0.9 |
| ETS | 0.67 | 0.64 | 0.80 | −0.1 |
| moving average (28) | 0.80 | 0.78 | 0.87 | −0.1 |
| seasonal naïve (last week) | 0.86 | 0.82 | 1.02 | −0.1 |
| mean | 0.96 | 0.96 | 0.99 | +3.3 |
| seasonal naïve (last year) | 1.04 | 1.03 | 1.08 | −0.1 |
| naïve / drift | 1.10 | 1.10 | 1.10 | +9.5 |

Table 4: Win rate and per-week improvement in RMSSE against the reference
benchmark and the stress check.

| model | vs. seasonal naïve: win / mean / median / Q1–Q3 | vs. 28-day mean: win / mean / median / Q1–Q3 |
|---|---|---|
| ARIMA | 82% / 20% / 24% / 10–37% | 75% / 14% / 17% / 0–34% |
| ETS | 83% / 19% / 21% / 8–33% | 71% / 12% / 14% / −3–32% |
| XGBoost | 75% / 16% / 23% / 0–40% | 72% / 11% / 16% / −3–33% |

The three models beat the reference in 75 to 83% of store-weeks by a median
of 21 to 24%, and the stress check in 71 to 75%. Their mean RMSSE is within
0.02, which is treated as a tie on ordinary weeks. What separates them is
the spread. XGBoost's median improvement equals ARIMA's, but its lower
quartile is zero against the reference and negative against the moving
average: in a quarter of weeks it does no better than the simplest thing.
The classical models are more consistent.

Which model is best depends on the store (Figure 3). XGBoost is best at the
three Texas stores and two of the three Wisconsin stores; ARIMA or ETS at
every California store. At the quietest store, 18 units a day, the moving
average is level with the classical models and ahead of XGBoost.

![Figure 3](figures/02_evaluate_4_rmsse_by_store.png)

Figure 3: Mean RMSSE by store, busiest first. Coloured lines are the models;
grey markers are the benchmarks. The line at 1.0 is seasonal naïve on the
training data.

It also depends on the week. On ordinary weeks the classical models are
slightly ahead. On holiday weeks XGBoost is best at seven of ten stores,
with a median improvement over the reference of 31% against ARIMA's 27% and
ETS's 21%, and ETS falls furthest. The gap between ARIMA and ETS on holiday
weeks, 0.74 against 0.80, is what the calendar regressors are worth, since
the two models are otherwise close. Figure 4 shows the eight weeks around
Thanksgiving and Christmas at the busiest store, where this happens.

![Figure 4](figures/02_evaluate_1b_forecasts_busiest_store_holidays.png)

Figure 4: Forecasts against actuals at the busiest store, 9 November 2015 to
3 January 2016. Black is what sold. Faint vertical lines are the Sunday
origins.

The lag-364 seasonal naïve, the holiday habit, is worse than the lag-7
version everywhere, holiday weeks included. The level drifted over the years
and last year's lookup carries the old level with it. Naïve and drift
over-forecast by nine units a day because every origin is a Sunday, the
busiest day, and they repeat it for the week.

## 7. Where XGBoost Loses

A good method's errors should be centred on zero and uncorrelated (FPP
§5.4). Figure 5 shows the mean error per store. ARIMA and ETS sit within
half a unit of zero at nine of ten stores. XGBoost over-forecasts by 3.9
units a day at the busiest store, about 4% of the level, and by 2.6 to 2.7
at three others, while under-forecasting by 3.6 at CA_2. This is not
visible in Table 3, where XGBoost's RMSSE is competitive; it only shows in
the bias column and in this figure. For an order it matters directly: a
model that runs 4% high every day is a model that over-orders every week at
that store.

![Figure 5](figures/02_evaluate_5_bias_by_store.png)

Figure 5: Mean forecast minus actual per store, busiest first. XGBoost's
markers sit above zero at seven of ten stores; the classical models' cluster
on the line.

The other places it loses are the quiet store, where there is too little
signal for the trees to improve on a moving average, and the lower quartile
of weeks generally. All three models fail the Ljung-Box test on their
one- to seven-step errors, which is expected for multi-step forecasts from
a shared origin; the diagnostic that would count against a method is a
spike at lag 7, and there is none. Error rises with horizon from about 0.64
at day 1 to about 0.78 at day 7 for all three, with the caveat that every
origin is a Sunday, so day 7 is always Sunday and the rise mixes horizon
with weekday.

## 8. Testing

The validator (`python -m src.validate`) runs seven checks: closed-form
benchmark answers on series whose correct forecast is known; every feature
recomputed from the raw panel by date filtering and compared to the
array-sliced implementation; per-state SNAP flags against the raw calendar;
the holiday-proximity counts and closure flags recounted by hand; a
synthetic noise floor no honest model can beat; a shuffled target no model
should learn from; and a byte-identical rerun.

The test suite (`python -m pytest tests`) was written by an
independent reviewer against the finished code and found four defects, all
fixed: the leakage assertion was defined but never called; the
feature-matrix cache key did not include the loader version; the by-store
table was sorted quietest-first under a "busiest first" label; and ARIMA's
chosen orders persisted across runs in one process. None of the four changed
a reported number, which was verified by checking the cached matrices
against the current panel.

## 9. Limitations and Future Work

- One item, all stores smooth. The demand-class routing is a method here,
  not a result, and is untested on an intermittent item.
- The horizon plot is confounded with weekday. Stepping the origins by a
  number coprime to seven would separate them.
- The ARIMA order is fixed for the year from the first training window.
- A point forecast is not an order. Ordering the forecast under-covers the
  week's demand about half the time, and the right order is a quantile at a
  service level set by the cost of running out against the cost of holding.
  That step is the subject of the follow-on repository,
  `demand-forecasting-order`.

Next, in order: train XGBoost directly on the quantile objective at a chosen
service level and compare it with ARIMA and ETS given the same treatment;
step the fold origins to fix the horizon confound; run the pooled model
across stores; and repeat the study on an intermittent item.

## Appendix: Reproducing the Results

```
python -m src.validate          # eight checks; must pass before any number is quoted
python -m pytest tests
python -m src.step5_evaluate    # the tables in section 6
python -m src.render_figures    # every figure, one folder per notebook
```

The notebooks `01_explore` and `02_evaluate` walk the same procedure with
the reasoning beside each cell. The dated answers to their checklists for
this item are in `findings/`.
