# Item 5 — XGBoost on a level-relative target

**Status: development numbers.** Weekly origins (52 folds, step 7), not the
reported every-day layout. The every-day confirmation (four runs of the new
method, about 2 h) has not been made. Nothing here is a reported result
until it has.

## Provenance

Every number below comes from these runs, made at commit `fda45fe` on
branch `item5-level-relative-target` on 2026-09-23 between 10:27 and 11:01.
The helper flagged the working tree as changed; the only change was an
ignore-file line, not code.

**FOODS_1_021, per store**
```
config: Config(cat_id=None, dept_id=None, fold_step=7, horizon=7, item_ids=('FOODS_1_021',), mask_holidays=False, min_train_days=365, n_folds=52, pool_by=None, rmsse_scale_lag=7, rmsse_scale_window='pre_holdout', season=7, seed=0, store_ids=(), test_window=7, use_price=False)
cache files:
  xgboost: xgboost_5f928f59ba7b.parquet
  xgboost_rel: xgboost_rel_9cfcfd899623.parquet
  ets: ets_7a864b86a3df.parquet
  moving_average_28: moving_average_28_1b40ebc9fb3c.parquet
  arima: arima_27823fdb0186.parquet
```

**FOODS_1_021, pooled**
```
config: as above with pool_by='item_id'
cache files:
  xgboost: xgboost_9a8d4d77e28f.parquet
  xgboost_rel: xgboost_rel_f5d19c9057b2.parquet
```

**FOODS_3_586, per store**
```
config: as the first block with item_ids=('FOODS_3_586',)
cache files:
  xgboost: xgboost_e3bc059475ec.parquet
  xgboost_rel: xgboost_rel_5b3d02fa6f0f.parquet
  ets: ets_5e4284a08a7a.parquet
  moving_average_28: moving_average_28_a89974eb4ecf.parquet
  arima: arima_e423a9f56935.parquet
```

**FOODS_3_586, pooled**
```
config: as the previous block with pool_by='item_id'
cache files:
  xgboost: xgboost_06e3637e23a9.parquet
  xgboost_rel: xgboost_rel_7d4b637dce36.parquet
```

Run times: FOODS_1_021 per store 1,272 s (every method fitted fresh),
pooled 112 s; FOODS_3_586 per store 384 s (only `xgboost_rel` fresh), pooled
260 s.

## The change

`xgboost_rel` keeps every feature and setting of the point model and changes
only the target: the trees learn sales minus the 28-day mean at the origin,
and the forecast is the prediction plus that mean. The 28-day mean is
computed at each row's own origin in training and at the fold origin in
prediction, so the target uses no data past the origin. Item 4 named the
failure this addresses: trees cannot extrapolate below the levels they were
trained on, so on a declining item their forecast stays where the history
was.

## FOODS_1_021 (intermittent, declining): the gap to the simple methods closes

| method | RMSSE all | normal | holiday | bias, units/day |
|---|---|---|---|---|
| 28-day moving average | **0.496** | 0.493 | 0.509 | +0.08 |
| XGBoost, level-relative, pooled | 0.498 | 0.495 | 0.513 | +0.15 |
| ETS | 0.499 | 0.496 | 0.514 | +0.11 |
| XGBoost, level-relative, per store | 0.506 | 0.503 | 0.519 | +0.17 |
| ARIMA | 0.518 | 0.510 | 0.552 | +0.31 |
| XGBoost, pooled | 0.571 | 0.564 | 0.605 | +0.56 |
| XGBoost, per store | 0.611 | 0.604 | 0.639 | +0.95 |

The per-store model moves from 0.611 to 0.506 and the pooled model from
0.571 to 0.498, level with ETS and 0.002 behind the moving average. At the
two stores that declined most, the over-forecast nearly disappears: CA_1
+3.65 → +0.82 units a day (per store), WI_1 +2.28 → +0.03. On an item whose
weekly pattern is this weak, the trees match the moving average; they do
not beat it.

## FOODS_3_586 (fast mover): no cost, a small gain

| method | RMSSE all | normal | holiday | bias, units/day |
|---|---|---|---|---|
| XGBoost, level-relative, pooled | **0.616** | 0.607 | 0.654 | +0.46 |
| XGBoost, pooled | 0.625 | 0.612 | 0.676 | +0.51 |
| XGBoost, level-relative, per store | 0.647 | 0.633 | 0.705 | +0.25 |
| ARIMA | 0.651 | 0.631 | 0.735 | −0.09 |
| XGBoost, per store | 0.665 | 0.653 | 0.717 | +0.88 |
| ETS | 0.674 | 0.643 | 0.802 | −0.14 |
| 28-day moving average | 0.797 | 0.778 | 0.873 | −0.07 |

The largest single gain is at CA_4, the quietest store: per store 0.719 →
0.588. The over-forecast at the busiest store remains: TX_2 +3.91 (plain,
per store) → +2.32 (level-relative, per store) → +1.85 (level-relative,
pooled).

## By store (RMSSE, weekly origins)

FOODS_1_021:

| store | xgboost | xgboost_rel | xgboost pooled | xgboost_rel pooled | ets | moving average |
|---|---|---|---|---|---|---|
| CA_1 | 0.691 | 0.496 | 0.509 | 0.492 | 0.514 | 0.485 |
| CA_2 | 0.661 | 0.651 | 0.651 | 0.656 | 0.643 | 0.664 |
| WI_1 | 0.694 | 0.416 | 0.525 | 0.409 | 0.422 | 0.410 |
| CA_4 | 0.611 | 0.517 | 0.553 | 0.500 | 0.494 | 0.491 |
| CA_3 | 0.558 | 0.542 | 0.542 | 0.534 | 0.536 | 0.538 |
| WI_2 | 0.714 | 0.580 | 0.644 | 0.575 | 0.570 | 0.576 |
| WI_3 | 0.550 | 0.508 | 0.526 | 0.486 | 0.481 | 0.490 |
| TX_2 | 0.543 | 0.405 | 0.545 | 0.402 | 0.393 | 0.393 |
| TX_1 | 0.627 | 0.613 | 0.772 | 0.602 | 0.621 | 0.602 |
| TX_3 | 0.458 | 0.333 | 0.447 | 0.325 | 0.319 | 0.312 |

FOODS_3_586:

| store | xgboost | xgboost_rel | xgboost pooled | xgboost_rel pooled | ets | moving average |
|---|---|---|---|---|---|---|
| TX_2 | 0.546 | 0.566 | 0.530 | 0.539 | 0.607 | 0.872 |
| CA_3 | 0.713 | 0.683 | 0.664 | 0.678 | 0.732 | 0.882 |
| TX_3 | 0.665 | 0.692 | 0.614 | 0.636 | 0.747 | 0.807 |
| TX_1 | 0.597 | 0.592 | 0.601 | 0.583 | 0.634 | 0.700 |
| CA_1 | 0.625 | 0.601 | 0.602 | 0.574 | 0.628 | 0.870 |
| CA_2 | 0.925 | 0.904 | 0.839 | 0.814 | 0.897 | 1.128 |
| WI_3 | 0.507 | 0.500 | 0.504 | 0.480 | 0.554 | 0.651 |
| WI_2 | 0.708 | 0.699 | 0.662 | 0.665 | 0.704 | 0.721 |
| WI_1 | 0.648 | 0.641 | 0.644 | 0.629 | 0.669 | 0.757 |
| CA_4 | 0.719 | 0.588 | 0.586 | 0.565 | 0.562 | 0.577 |

On the fast mover the level-relative target is worse at the busiest stores
and better at the rest. Per store it is worse at TX_2 and TX_3 and better at
the other eight. Pooled, it is worse at the three busiest (TX_2, CA_3, TX_3)
and at WI_2 (by 0.003), and better at the other six.

## Bias by store (units a day, forecast minus actual)

| store | FOODS_1_021 xgboost | pooled | rel | rel pooled | FOODS_3_586 xgboost | pooled | rel | rel pooled |
|---|---|---|---|---|---|---|---|---|
| CA_1 | 3.65 | 1.10 | 0.82 | 0.60 | 2.62 | 1.35 | 1.03 | 1.17 |
| CA_2 | −0.15 | −0.14 | −0.07 | 0.06 | −3.56 | −1.48 | −2.06 | −0.98 |
| CA_3 | 0.44 | 0.35 | 0.07 | 0.09 | −0.29 | 0.60 | −0.25 | 0.76 |
| CA_4 | 1.20 | 0.74 | 0.27 | 0.13 | 2.71 | 0.88 | 0.04 | 0.19 |
| TX_1 | 0.11 | 0.54 | 0.07 | 0.12 | 0.40 | 0.41 | −0.22 | 0.51 |
| TX_2 | 0.57 | 0.66 | 0.12 | 0.12 | 3.91 | 2.11 | 2.32 | 1.85 |
| TX_3 | 0.49 | 0.58 | 0.13 | 0.11 | 2.65 | 1.38 | 1.49 | 1.35 |
| WI_1 | 2.28 | 1.09 | 0.03 | 0.09 | 0.10 | 0.09 | 0.08 | −0.00 |
| WI_2 | 0.41 | 0.28 | 0.15 | 0.12 | −0.87 | −0.47 | −0.33 | −0.20 |
| WI_3 | 0.44 | 0.44 | 0.16 | 0.09 | 1.16 | 0.24 | 0.41 | −0.07 |

## What it says, pending confirmation

1. One tree configuration, pooled and level-relative, is best on the fast
   mover and level with the best simple method on the declining
   intermittent item. Before this change the learned model had to be routed
   away from the second item.
2. The mechanism named in item 4 is confirmed by the fix: removing the
   level from the target removes most of the bias on the declining item.
3. The residual over-forecast at the busiest fast-mover stores is not a
   level problem and is not fixed by this change.

## Not done

- The every-day confirmation: `xgboost_rel` per store and pooled, both
  items, `Config(fold_step=1, n_folds=358)`. Everything else those tables
  need is cached.
- Win rates and improvement quartiles against the benchmarks.
