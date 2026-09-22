# Does a learned model beat the orderer's screen? Daily grocery demand, ten stores, one year of forecasts

*A forecaster's-toolbox study on public data, built along the five steps of*
Forecasting: Principles and Practice *(Hyndman & Athanasopoulos).*
Draft of 2026-09-22. Code and tests: the repository this file lives in.

---

## The question, and the short answer

A store orders a fast-moving grocery item every week. The default screen an
orderer sees is *this day last week*; before a holiday they look at *this day
last year*. Does a gradient-boosted model beat that, by how much, and does the
answer depend on the store or on the week?

On one fast-moving item at ten stores, scored on every week of a full year:

- **Three learned models, XGBoost, exponential smoothing (ETS) and seasonal
  ARIMA, all beat every simple method at every store**, cutting scaled error
  by about a quarter against the orderer's default screen. They are within
  0.02 of each other over the year: on ordinary weeks a tie.
- **Which one wins depends on the store and on the week.** XGBoost leads at
  the Texas stores and on holiday weeks; the classical models lead at the
  California stores and on ordinary weeks. ETS, the one model that cannot be
  told a holiday is coming, falls furthest on holiday weeks.
- **A point forecast is not an order.** Ordering the forecast stocks out half
  the weeks, measured. Turned into a quantile order at a chosen service
  level, the model ranking *changes with the cost of being wrong*: XGBoost
  ties ARIMA where running out is the expensive mistake and falls to fifth
  where holding is. A symmetric error metric could not have shown that.

Everything reported here passed an eight-check validator and an 82-test
suite before it was quoted, and every table can be regenerated from the
repository in under a minute from cache.

---

## 1. Defining the problem

Replenishment consumes a forecast **one day at a time**: the store runs out on
a Saturday, not "this week". So the forecast is daily, seven days ahead from
a single origin, exactly as an order is placed: stand on Sunday, order for
Monday through Sunday. In the book's notation, ŷ<sub>T+h|T</sub> for
h = 1…7.

Two consequences shape everything after. Because the forecast feeds an order,
**bias matters in its own right**: a method that is 5% low every day turns
into stockouts, while a noisy but centred one only turns into safety stock.
And because the order is a weekly quantity at a service level, the point
forecast is only the first half of the answer; section 6 supplies the second.

The data is the public M5 set: daily unit sales for Walmart stores in three
US states, 2011–2016, with a calendar of events, SNAP benefit days and shelf
prices. The item studied, `FOODS_3_586`, was chosen by the availability
screen in step 3: continuously stocked at all ten stores, selling 14 to 103
units a day depending on the store. That volume gradient is what the
comparison is run along.

## 2. Gathering the data, and two things the loader does

The three raw files are reshaped to one row per store-item-day and joined to
the calendar, with each row taking the SNAP flag for its own state. Two
decisions live in the loader rather than downstream, so nothing can see the
raw version by accident.

**A closure is a missing value, not a zero.** Every store records zero on
Christmas Day because the stores were shut. Left as zero it does damage well
beyond the day: the seasonal-naïve forecast for the following week reads it,
every rolling mean is dragged down for a week, and smoothing methods take it
as a level shock. Following the book's treatment of missing values (§13.7)
the day is imputed with the same-weekday mean of the surrounding weeks,
flagged, and excluded from every score.

**Which holidays matter is measured, not assumed.** Each of the calendar's 30
events was scored against a same-weekday baseline, on the day and on the two
days before it. Seven move this item by more than 15%: Thanksgiving (1.7×
on the day), Christmas (1.7× two days before, 1.5× the day before, then
shut), Labor Day (1.4×), Independence Day and Valentine's Day (1.2× with a
1.3× run-up), Easter (1.2× run-up) and New Year (0.85×). The other 23 sit
within a few percent of baseline. Those seven drive three features known
years in advance: a flag, days to the next major event, and days since the
last. An on/off flag alone cannot learn the run-up before Christmas, which in
this data is larger than most holidays' own day.

![Calendar events: how much each one moves sales](figures/01_explore_6_event_effects.png)

## 3. Looking before modelling

The book is blunt: *always start by graphing the data.* Four things came out
of it.

**The demand is smooth, everywhere.** The standard classification for
whether a series is a candidate for a learned model at all is two numbers:
the average demand interval (how often it sells) and the squared coefficient
of variation of the sale sizes (how consistent the amounts are). Cut at the
conventional thresholds they give four classes: smooth, erratic,
intermittent, lumpy. Every store of this item is smooth: it sells every day
(ADI 1.00–1.01) with sale sizes varying modestly (CV² 0.06–0.24 against a
cut of 0.49). That is a fact about the item, and it decides which question
the study can answer: not "does the best method change with demand class",
but "which method wins on a fast mover, and does it depend on the store".
The method is the reusable part: an intermittent item would be routed to
different methods and different metrics, and the classification is computed
on training data only, so no label is informed by a scored day.

![Demand classification](figures/01_explore_2_classmap.png)

**The structure is weekly, on top of a drifting level.** Weekend days sell
about 50% more than midweek at every store once each is indexed to its own
mean; the autocorrelation peaks at lags 7, 14, 21 and 28 with lag 1 strong
in its own right, and lags 6 and 8 nearly as tall as 7 (a broad weekend).
Those are the lags the models were given, chosen from the evidence rather
than by convention. An annual shape exists (August high, January low) on top
of a multi-year decline: the busiest store averaged 109 a day in 2011 and 82
in early 2016. The years share a shape, not a level.

![One product, every store](figures/01_explore_1_grid.png)

![Seasonal shape across stores](figures/01_explore_3b_season_all_stores.png)

**SNAP days carry signal, price does not.** Sales on benefit days run 6%
above ordinary days pooled, and positive at nine of ten stores; the dates
are fixed by state and known years ahead, so the effect is free forecasting
signal. Price, by contrast, took three values in five years, changing on the
same dates at every store and never inside a forecast window: a clock, not a
variable. Offered to the model it would teach "which era is it", which does
not transfer to a product whose price moves. It was left out, and switching
it on for XGBoost confirmed the call (scaled error 0.674 against 0.665).

## 4. The methods, and why there are six benchmarks

A single benchmark has misled this project once: an earlier version reported
a 90% win rate that was wrong twice over, once because lag features reached
into the test window, and once because the only benchmark was one a dull
model beats without skill. The rebuilt study reports six, each answering a
different "compared to what":

| benchmark | what it encodes |
|---|---|
| seasonal naïve (lag 7) | the orderer's default screen: this day last week |
| seasonal naïve (lag 364) | the orderer's holiday habit: this day last year |
| 28-day moving average | the recent level, ignoring the weekday |
| mean | the whole history's level |
| naïve, drift | the last value, flat and with a trend |

Seasonal naïve stakes everything on one past day, so on a noisy series its
error runs about √2 worse than predicting a mean, and a model can "beat" it
by being sensibly dull. The moving average is the benchmark that actually
competes.

Three learned models sit above them. **XGBoost** on lag, rolling, same-weekday
and calendar features. **ETS** (Holt-Winters with weekly seasonality), the
book's first classical recommendation for this kind of data. **Seasonal
ARIMA with regressors** for the holiday flag, the run-up and SNAP, so a
classical model gets the same calendar information XGBoost gets. A plain
ARMA is not in the set on purpose: it is ARIMA without the seasonal term,
and on daily data with a weekly cycle it would need absurd orders to imitate
what one seasonal term captures.

Every feature for a target day is computable from data available at the
origin, and that is asserted mechanically rather than trusted: training rows
are filtered by *when their answer became observable*, and a date comparison
refuses anything that reaches past the origin.

## 5. Evaluating on a full year

Every method is run through **52 walk-forward folds** of seven days, 25 May
2015 to 22 May 2016, one full year so that every season is scored once. The
book prescribes no fold count; its only sizing guidance is a test set of
about 20% of the sample, which one year is. Each fold trains on all history
before its Sunday origin and is scored on the following Monday to Sunday.

The score is **RMSSE**, the root mean squared error scaled by what a seasonal
naïve forecast would have made on the training data (§5.8). A value of 1.0
means "no better than repeating last week's same weekday"; 0.8 means 20%
better; and because the denominator belongs to the series and not to the
method, a 100-unit store and a 15-unit store are comparable.

**Every table is reported twice**, on ordinary weeks and on the weeks that
touch a major holiday, because a single pooled number cannot say whether a
method's advantage came from the fifty ordinary weeks or from the handful
where the calendar does the work.

| method | all weeks | normal | holiday | bias (units/day) |
|---|---|---|---|---|
| ARIMA | **0.65** | **0.63** | 0.74 | −0.1 |
| XGBoost | 0.67 | 0.65 | **0.72** | +0.9 |
| ETS | 0.67 | 0.64 | 0.80 | −0.1 |
| moving average (28) | 0.80 | 0.78 | 0.87 | −0.1 |
| seasonal naïve (last week) | 0.86 | 0.82 | 1.02 | −0.1 |
| mean | 0.96 | 0.96 | 0.99 | +3.3 |
| seasonal naïve (last year) | 1.04 | 1.03 | 1.08 | −0.1 |
| naïve / drift | 1.10 | 1.10 | 1.10 | +9.5 |

What the year says:

- **The models are separable from the benchmarks, not from each other.**
  All three beat the strongest benchmark by 0.12–0.15 and the default screen
  by about 0.2, and beat seasonal naïve in 75–83% of store-weeks. Among
  themselves the medians sit within 0.04: a tie on ordinary weeks, and the
  report says so rather than crowning one.
- **The store decides.** XGBoost leads at the three Texas stores and two of
  three Wisconsin stores; ARIMA or ETS at every California store. At the
  quietest store (18 a day) the 28-day mean is level with the classical
  models and ahead of XGBoost: the learned models earn their keep where there
  is signal to learn.
- **The week decides too.** On ordinary weeks the classical models edge
  XGBoost. On holiday weeks XGBoost is best at seven of ten stores and ETS
  falls furthest. The gap between ARIMA and ETS on holiday weeks, 0.74
  against 0.80, is the measured value of being told a holiday is coming.
- **"This day last year" is a worse guide than "this day last week", even in
  holiday weeks.** The level drifted, and the annual lookup carries the old
  level with it. The orderer's holiday habit is right about the shape and
  wrong about the size.
- **Naïve and drift over-forecast by nine units a day** because every origin
  is a Sunday, the busiest day, and they repeat it for the week.

![RMSSE by store](figures/02_evaluate_4_rmsse_by_store.png)

![Forecasts against actuals at the busiest store](figures/02_evaluate_1_forecasts_busiest_store.png)

**Residuals.** The book asks that a good method's errors be centred on zero
and uncorrelated (§5.4). Over the year at the busiest store ARIMA and ETS are
centred (+0.4 units a day); XGBoost is not (+3.9, about 4% of the level),
and its over-forecast recurs at three other stores while ARIMA and ETS stay
within half a unit at nine of ten. That is XGBoost's one systematic
weakness here, and section 6 corrects it automatically. All three show lag-1
autocorrelation of 0.3–0.4, which is expected rather than a defect: the
book's test is stated for one-step residuals, and these are one- to
seven-step errors from a shared origin, correlated out to lag 6 even for a
perfect model. What would count against a method is a spike at lag 7, and
there is none.

**Horizon.** Error rises from h = 1 to h = 7 for every method, the models
from 0.64 to about 0.78. Read with one caveat: every origin is a Sunday, so
day 6 is always Saturday and day 7 always Sunday, the two busiest days, and
the plot mixes "how far ahead" with "which weekday".

![Error by horizon](figures/02_evaluate_2_rmsse_by_horizon.png)

## 6. From forecast to order

Everything above scores a point forecast with a symmetric error. A
replenishment decision needs two things it does not have.

**The order is a weekly total.** A Sunday order covers Monday to Sunday, so
the error that reaches the shelf is the week's forecast total minus the
week's actual total; daily errors partly cancel inside a week and the daily
score never sees that they did.

**Ordering the mean stocks out half the time.** A point forecast is the
centre of what might happen; demand lands above it about as often as below.
Measured on this year: ARIMA, ETS and the benchmarks under-forecast the
weekly total in 50% of weeks, XGBoost in 44% (its bias buys six units a
week of cover). The quantity to order is a *quantile*, the level that covers
demand with a chosen probability τ, and which τ is an economic question, not
a statistical one: the understock cost over the sum of both costs. Above 0.5
when running out is the expensive mistake (ambient grocery, typically near
0.9); below 0.5 when holding is (fresh produce that goes in the bin, near
0.3). This item is anonymised, so its τ is unknown, and the study reports a
range and asks whether the answer changes across it.

**Where the quantiles come from.** None of the nine methods produces a
distribution, so all nine are given one the same way (§5.5): a method's own
weekly errors over a *calibration year* (May 2014 to May 2015) are its
uncertainty, and its τ-quantile forecast is the point forecast plus the
τ-quantile of those errors. A method with tighter errors earns a smaller
add-on; a biased method is corrected automatically, because the mean error
is inside the distribution being shifted by. The calibration year precedes
the scored year, so no quantile is judged on the weeks it was estimated from.

**The metric is pinball loss**, the book's quantile score (§5.9): a unit of
shortfall costs 2τ, a unit of surplus 2(1 − τ). At τ = 0.9 running out is
nine times as expensive as over-ordering. The asymmetry the decision cares
about is inside the metric.

Relative pinball loss (÷ mean weekly sales; lower is better; compare within
a column, never along a row):

| method | τ = 0.3 | τ = 0.5 | τ = 0.7 | τ = 0.9 | coverage at 0.9 |
|---|---|---|---|---|---|
| ARIMA | **0.094** | **0.106** | **0.096** | 0.053 | 0.92 |
| XGBoost | 0.114 | 0.116 | 0.104 | **0.052** | 0.92 |
| moving average (28) | 0.105 | 0.117 | 0.107 | 0.062 | 0.90 |
| ETS | 0.104 | 0.120 | 0.107 | 0.059 | 0.89 |
| seasonal naïve | 0.109 | 0.123 | 0.110 | 0.061 | 0.89 |

- **The ranking changes with the cost asymmetry.** At τ = 0.9, an ambient
  item's service level, XGBoost and ARIMA tie. At τ = 0.3, a perishable's,
  XGBoost falls to fifth, behind two benchmarks. RMSSE said nothing about
  this and could not: a symmetric metric cannot see an asymmetric cost, and
  which way it misleads depends on economics it has no access to.
- **ARIMA is the most robust method** across the whole range and the best on
  holiday weeks at every τ.
- **The quantiles transferred.** A 0.9 order covered 89–92% of weeks for
  every model; one year's errors described the next well enough to order
  from. The plain mean is the exception (83%): its bias is not stable across
  years.
- **XGBoost's calibration transfers least well**, because its errors shrink
  as training data grows, so last year's spread overstates this year's.
  Letting the calibration window grow through the scored year, as a live
  system would, recovers most of the gap (0.099 at τ = 0.3, second place).

![Quantile score and coverage by service level](figures/03_order_1_pinball_and_coverage.png)

![Weekly order-up-to levels at the busiest store](figures/03_order_2_weekly_order_band.png)

At the busiest store, ARIMA's τ = 0.9 order sat a constant 116 units a week
above its median forecast, and demand rose above it in one week of 52. The
January trough is the opposite failure, a month where every method's median
sat above what sold. Neither is a bug; both are what a service level means.

## 7. How this maps onto a store's ordering

Three facts about how ordering actually works shaped the design, and each
has a counterpart in the results.

**The default screen is the benchmark.** An orderer's default view is this
day last week; before a holiday, this day last year. Both are in the
comparison as benchmarks, so "the model beats the benchmark" means "the
model beats what a person does by default", and the holiday-week column is
the model against the expert override. The override, it turns out, is right
about the shape of a holiday and wrong about the level.

**Deliveries are known, on-hand is inferred.** A store system typically sees
what the warehouse shipped and what the tills sold, not a running on-hand
count. On-hand is therefore cumulative deliveries minus cumulative sales
minus shrink, re-anchored by a periodic physical count, and it drifts between
counts. Two consequences for the order: it must subtract what was *shipped*,
not what was ordered, since the warehouse can short a line and the store
learns of it only when the truck arrives; and the safety margin has to cover
inventory-record error as well as demand uncertainty. Orders can also be
placed on consecutive days before the first arrives, so the cover period is
the lead time plus the review period, not the lead time alone.

**Sales are censored demand.** When the shelf is empty, sales understate
demand, the forecast drifts low, and the next order drifts lower: a loop.
The availability screen guards against it for this item (longest zero run
10 days, no stocking gaps), and delivery data gives a stronger guard than
sales alone, since implied on-hand reaching zero identifies exactly which
days are censored. M5 has no deliveries, so the replenishment step here is a
calculation on demand, not a simulation of a shelf, and the report says so.

## 8. What is deliberately not claimed, and what comes next

- **One item.** Every store is smooth, so the demand-class discussion is a
  method, not a result, for this item. The pipeline routes an intermittent
  item to different methods and metrics; that routing is wired and untested
  on a real intermittent series.
- **No inventory positions.** Receipts, on-hand and shrink are absent from
  the data, so the order quantity is the demand side of the decision only.
- **The service level is a range, not a number**, because the item's
  economics are unknown. For a real product τ is set once from the margin
  and the write-off cost, and the pinball column at that τ is the single
  number to optimise.
- **The ARIMA order is chosen once per store** on its first training window
  and held for the year, by information criterion at fixed differencing
  (§9.7). The two-year run used for the quantile section chooses on a window
  a year earlier than the one-year run, so its ARIMA point forecasts differ
  slightly from section 5's; every other method is identical between the two.
- **The horizon plot is confounded with weekday**, as stated in section 5.
- **Cold start is unsolved here.** A new product has no history to
  calibrate from. The practical routes are pooling across stores of the same
  item (wired but not run here), borrowing the seasonal shape and error
  distribution of an analogous item until history accumulates, and, more
  recently, pre-trained time-series foundation models that forecast from a
  short history without fitting. Each is a project of its own; none is
  claimed.

The natural next steps are, in order: train XGBoost directly on the quantile
objective at the chosen τ rather than calibrating a point forecast after the
fact; run the pooled model across stores; and repeat the study on an
intermittent item, where the classification would route to different tools.

## Reproducing this

```
python -m src.validate          # eight checks; must pass before any number is quoted
python -m pytest tests          # 82 tests
python -m src.step5_evaluate    # the point-forecast tables
python -m src.order             # the order-quantity tables (two-year run)
python -m src.render_figures    # every figure, one folder per notebook
```

The three notebooks (`01_explore`, `02_evaluate`, `03_order`) walk the same
procedure with the reasoning beside each cell, and `findings/` holds the
dated answers to their checklists for this item.
