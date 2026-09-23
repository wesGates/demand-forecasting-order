# Item 4 — an intermittent, declining item

## Provenance

Every number below comes from these runs. The branch and commit name the code; the config line rebuilds the run; the cache files hold the predictions, each with a sidecar recording the config and code digest it was made under.

**per-store run**
```
branch: item4-intermittent
commit: 391ff3f (working tree has uncommitted changes)
config: Config(cat_id=None, dept_id=None, fold_step=1, horizon=7, item_ids=('FOODS_1_021',), mask_holidays=False, min_train_days=365, n_folds=358, pool_by=None, rmsse_scale_lag=7, rmsse_scale_window='pre_holdout', season=7, seed=0, store_ids=(), test_window=7, use_price=False)
cache files:
  mean: mean_053cde422719.parquet
  naive: naive_a885bc599512.parquet
  seasonal_naive: seasonal_naive_4ca182af4267.parquet
  seasonal_naive_364: seasonal_naive_364_6f78c782fe4b.parquet
  drift: drift_6898c99decf7.parquet
  moving_average_28: moving_average_28_7b9eb34bcfda.parquet
  xgboost: xgboost_52b5d521ded9.parquet
  ets: ets_41074048d94c.parquet
  arima: arima_3466473e044b.parquet
```

**pooled run**
```
branch: item4-intermittent
commit: 391ff3f (working tree has uncommitted changes)
config: Config(cat_id=None, dept_id=None, fold_step=1, horizon=7, item_ids=('FOODS_1_021',), mask_holidays=False, min_train_days=365, n_folds=358, pool_by='item_id', rmsse_scale_lag=7, rmsse_scale_window='pre_holdout', season=7, seed=0, store_ids=(), test_window=7, use_price=False)
cache files:
  xgboost: xgboost_4d724e78629a.parquet
```

Dated 2026-09-23. `FOODS_1_021`, ten stores, every-day layout (358 origins,
3,580 store-origins per method). Chosen by the classification, not by hand:
of 216 items in one department, one of two whose ten stores span three
demand classes with no zero run over 30 days. Volumes 0.6–7.2 units a day;
smooth at WI_1, erratic at CA_1, CA_4 and CA_2, lumpy at the other six.
Item sales fell from 2.8 units a day (2014) to 1.25 (early 2016). Run time
1 h 38 min for the point methods; 15 min for the pooled model.

## The simple methods win, and the classification said they would

| method | RMSSE all | normal | holiday | bias, units/day | win vs 28-day mean | median improvement |
|---|---|---|---|---|---|---|
| 28-day moving average | **0.497** | 0.496 | 0.500 | +0.08 | | |
| ETS | 0.500 | 0.499 | 0.504 | +0.14 | 47% | −1.0% |
| ARIMA | 0.518 | 0.511 | 0.549 | +0.33 | 42% | −3.3% |
| XGBoost, pooled | 0.570 | 0.564 | 0.596 | +0.57 | 36% | −6.7% |
| XGBoost, per store | 0.610 | 0.605 | 0.630 | +0.95 | 29% | −10.4% |
| mean | 0.614 | 0.613 | 0.619 | +0.97 | | |
| seasonal naïve | 0.687 | 0.691 | 0.670 | +0.03 | | |

By demand class (mean RMSSE):

| class | stores | moving average | ETS | ARIMA | XGBoost pooled | XGBoost per store |
|---|---|---|---|---|---|---|
| smooth | 1 | 0.407 | 0.417 | 0.438 | 0.519 | 0.687 |
| erratic | 3 | 0.550 | 0.553 | 0.576 | 0.571 | 0.657 |
| lumpy | 6 | 0.486 | 0.487 | 0.503 | 0.578 | 0.573 |

XGBoost loses at every one of the ten stores, per store or pooled. The
28-day moving average or ETS is best at nine; ARIMA ties at CA_2 and CA_3.
Neither ETS nor ARIMA fell back to the safety forecast on any fold.

## Why XGBoost loses here: it cannot follow a falling level

The item declined through the scored year. Mean daily sales and mean daily
forecast by month:

| | May 15 | Aug 15 | Nov 15 | Feb 16 | May 16 |
|---|---|---|---|---|---|
| actual | 2.72 | 1.85 | 0.93 | 1.25 | 1.09 |
| ETS | 3.13 | 2.36 | 1.31 | 1.25 | 1.01 |
| moving average | 2.92 | 2.12 | 1.25 | 1.29 | 0.98 |
| XGBoost, pooled | 2.88 | 2.70 | 1.89 | 1.72 | 1.41 |
| XGBoost, per store | 2.95 | 2.76 | 2.21 | 2.46 | 1.92 |

The smoothing methods have a level state that moves with the data. A tree
ensemble predicts from leaves grown on training-era levels and cannot
extrapolate below them, so as the item declines its forecast stays where
the history was. Over the year the per-store model's mean forecast is 2.54
against an actual of 1.58, 60% high; it exceeds the actual on three days
in four; on the 46% of days with zero sales it forecasts 2.05. Pooling
across stores helps (2.16, bias halved, RMSSE 0.610 → 0.570) because the
ten stores' shared recent level pulls the leaves down, but it does not
close the gap: the best simple method is still 0.07 better.

The over-forecast concentrates at the two highest-volume stores, which
declined the most: CA_1 (+3.7 units a day per store, +1.6 pooled) and WI_1
(+2.3, +1.1).

## What it says

1. **The routing works.** The classification put this item in the classes
   where a learned model was expected to lose, and it lost at every store.
   On the fast mover (items 2 and 3) it won at eight of ten. The two items
   together are the case for routing by demand type rather than one model
   for everything.
2. **The failure is specific, not vague.** The earlier analysis reported
   that the learned model does worse on "sporadic and inconsistent"
   series. Here the mechanism is named and measured: level drift on a
   low-volume series, which trees cannot extrapolate and which
   exponential smoothing tracks by construction. Sparsity alone is not the
   cause; the smooth store loses by the most.
3. **Bias, again.** RMSSE puts pooled XGBoost 0.07 behind the moving
   average; the bias column puts it 0.57 units a day high on an item
   selling 1.6. For an order that is a 36% over-order, every week.
4. **Which simple method.** On this item the 28-day moving average is the
   best forecast of all, and the seasonal naïve is among the worst: the
   weekly pattern is weak at these volumes and the recent level is
   everything. The right benchmark depends on the demand class too.

## What would fix it, not done here

- A level-aware target for the trees: forecast the *deviation from the
  recent level* (the difference from the 28-day mean, say) rather than
  the level itself, so extrapolation is about the residual shape, not the
  level. This is the natural next experiment for the learned model on
  declining items, and it is a feature-engineering change, not a new
  model.
- Count-aware objectives (Poisson or Tweedie) for the tree model on
  low-volume series.
- Classical intermittent-demand methods (Croston, TSB) for the lumpy
  class, which are wired nowhere yet.

## Bookkeeping

124 of 3,580 store-weeks had a seasonal naïve score of exactly zero (a
week of zeros forecast as zeros), and 23 had a zero moving-average score.
The improvement table now excludes those from the quartiles and reports
the count, since a percentage improvement over zero is undefined.
