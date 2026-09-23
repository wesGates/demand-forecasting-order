# Item 2 — every day an origin

## Provenance

Every number below comes from these runs. The branch and commit name the code; the config line rebuilds the run; the cache files hold the predictions, each with a sidecar recording the config and code digest it was made under.

**run**
```
branch: item4-intermittent
commit: 391ff3f (working tree has uncommitted changes)
config: Config(cat_id=None, dept_id=None, fold_step=1, horizon=7, item_ids=('FOODS_3_586',), mask_holidays=False, min_train_days=365, n_folds=358, pool_by=None, rmsse_scale_lag=7, rmsse_scale_window='pre_holdout', season=7, seed=0, store_ids=(), test_window=7, use_price=False)
cache files:
  mean: mean_7af186ba56d9.parquet
  naive: naive_4a269bf6f92f.parquet
  seasonal_naive: seasonal_naive_fccb0fe81a6f.parquet
  seasonal_naive_364: seasonal_naive_364_f77d4df4aaff.parquet
  drift: drift_84c28e41c2c1.parquet
  moving_average_28: moving_average_28_78a6f14743a5.parquet
  xgboost: xgboost_9523523e8a9c.parquet
  ets: ets_addfb1ac6481.parquet
  arima: arima_7da8f684e059.parquet
```

Dated 2026-09-23. `FOODS_3_586`, ten stores, the same scored year
(25 May 2015 – 22 May 2016), `Config(fold_step=1, n_folds=358)`: 358 origins
one day apart, seven-day windows that overlap, 3,580 store-origins scored
per method against 520 on the tiled layout. Run time 2 h 11 min for the
point methods with a second job on the machine for part of it.

## The tables move very little

Mean RMSSE, tiled layout (step 7) against every-day layout (step 1):

| method | all: 7 → 1 | normal: 7 → 1 | holiday: 7 → 1 | bias: 7 → 1 |
|---|---|---|---|---|
| ARIMA | 0.651 → 0.647 | 0.631 → 0.622 | 0.735 → 0.756 | −0.09 → −0.01 |
| XGBoost | 0.665 → 0.666 | 0.653 → 0.649 | 0.717 → 0.742 | +0.88 → +0.79 |
| ETS | 0.674 → 0.670 | 0.643 → 0.635 | 0.802 → 0.820 | −0.14 → −0.07 |
| moving average (28) | 0.797 → 0.795 | 0.778 → 0.771 | 0.873 → 0.901 | |
| seasonal naïve | 0.861 → 0.864 | 0.822 → 0.825 | 1.024 → 1.034 | |
| naïve / drift | 1.099 → 1.002 | 1.098 → 0.978 | 1.101 → 1.104 | +9.5 → 0.0 |

Win rate and improvement against seasonal naïve: ARIMA 82% → 83%, median
24.2% → 25.4%; ETS 83% → 84%, 21.0% → 21.7%; XGBoost 75% → 76%, 22.9% →
23.1%, lower quartile 0.1% → 0.9%. Against the 28-day mean the win rates
are unchanged at 75 / 71 / 72%.

Holiday folds are 670 of 3,580 (18.7%), against 100 of 520 (19.2%).

## What changed, and why it is a correction

1. **Naïve and drift lost their nine-unit over-forecast.** On the tiled
   layout every origin was a Sunday, the busiest day, and "repeat the last
   value" repeated Sunday for the week. With origins rotating through the
   week the bias is zero and their RMSSE drops from 1.10 to 1.00. The
   earlier number was an artefact of the layout, not a property of the
   method.
2. **The horizon plot is monotone.** RMSSE by days ahead, models: ARIMA
   0.657 → 0.710, ETS 0.678 → 0.738, XGBoost 0.701 → 0.719 from h = 1 to
   h = 7. Seasonal naïve is flat at 0.93, as it should be (it uses the same
   lag at every horizon). The weekday confound is gone and the caveat in
   the parent's report can be deleted for this layout.
3. **The conclusions hold on every order day.** RMSSE by the weekday of the
   origin runs 0.691–0.703 for ARIMA, 0.716–0.734 for ETS and 0.706–0.714
   for XGBoost: flat. A Tuesday order and a Sunday order get the same
   answer.
4. **One-step residuals give a real Ljung-Box verdict.** At the busiest
   store, 357 one-step errors: ARIMA mean +0.12, lag-1 autocorrelation
   0.04, p = 0.17 (white noise); ETS −0.08, 0.16, p < 0.001; XGBoost +3.38,
   0.28, p < 0.001. ARIMA's one-step errors are clean. XGBoost's are both
   biased and autocorrelated at lag 1, which says it is under-using the
   most recent day even though yesterday's sales are a feature.

## Two readings worth a sentence

- **XGBoost's error barely grows with horizon** (0.701 to 0.719) while the
  classical models' grows by about 0.05. XGBoost is the worst of the three
  one day ahead and level with them a week ahead. It forecasts the week's
  shape as well as anyone and the next day worse than anyone.
- **Best method per store shifted toward ARIMA**: ARIMA leads at six
  stores on this layout (TX_3, TX_1, CA_1, CA_2, CA_3, WI_2), XGBoost at
  three (TX_2, WI_3, WI_1), ETS at CA_4. On holiday weeks XGBoost leads at
  six. The store-dependence finding stands; the tally is less flattering to
  XGBoost when every weekday is an origin.

## Caveat that belongs in any report using this layout

Adjacent folds share six of their seven scored days, so the 3,580
store-origins are not 3,580 independent samples. Win rates and quartiles
are exact descriptions of what happened, not seven times the evidence.

## Decision

Adopt `fold_step=1` as the reported layout. Keep step 7 and the `DEV`
preset for iteration. The parent's report is not reissued; the follow-on
reports from this layout.

## Figures

![horizon](figures/item2_rmsse_by_horizon_step1.png)

![arima one-step residuals](figures/item2_residuals_arima_step1.png)
