# Item 3 — one XGBoost for all ten stores

Dated 2026-09-23. `FOODS_3_586`, every-day layout (358 origins, 3,580
store-origins scored), `Config(pool_by="item_id")`: one model per origin
trained on all ten stores' rows with store identity as a categorical
feature, against the per-store models of item 2. Features, early stopping
and clipping identical. Run time 16 min for 358 fits of about 119,000 rows,
after the harness was corrected to fit once per origin rather than once per
store per origin.

## The pooled model is the best method in the study

| method | RMSSE all | normal | holiday | bias, units/day | win vs seasonal naïve | median improvement | Q1 | Q3 |
|---|---|---|---|---|---|---|---|---|
| **XGBoost, pooled** | **0.621** | **0.607** | **0.681** | +0.25 | **84%** | **26.9%** | **9.6%** | 40.8% |
| ARIMA | 0.647 | 0.622 | 0.756 | −0.01 | 83% | 25.4% | 9.4% | 38.1% |
| XGBoost, per store | 0.666 | 0.649 | 0.742 | +0.79 | 76% | 23.1% | 0.9% | 39.0% |
| ETS | 0.670 | 0.635 | 0.820 | −0.07 | 84% | 21.7% | 7.4% | 34.3% |

Against its per-store counterpart, store-origin by store-origin: the pooled
model wins 62% of the time, with a median gain of 5.3% (quartiles −6.5% to
+17.8%).

## Where the gain comes from

By store, busiest first (RMSSE, per-store → pooled):

| store | units/day | per-store XGBoost | pooled XGBoost | ARIMA |
|---|---|---|---|---|
| TX_2 | 103 | 0.546 | 0.530 | 0.577 |
| CA_3 | 70 | 0.709 | 0.662 | 0.693 |
| TX_3 | 80 | 0.672 | 0.609 | 0.667 |
| TX_1 | 60 | 0.609 | 0.598 | 0.604 |
| CA_1 | 47 | 0.626 | 0.595 | 0.601 |
| CA_2 | 31 | 0.930 | 0.826 | 0.875 |
| WI_3 | 48 | 0.507 | 0.503 | 0.535 |
| WI_2 | 14 | 0.708 | 0.659 | 0.672 |
| WI_1 | 17 | 0.650 | 0.648 | 0.676 |
| CA_4 | 18 | 0.706 | 0.583 | 0.572 |

1. **It helps most where the per-store model was weakest.** CA_4, the
   quietest store, goes from 0.706 (behind a moving average) to 0.583
   (level with ETS and ARIMA); CA_2 from 0.930 to 0.826; CA_3 and TX_3 by
   0.05–0.06. The busy Texas stores, where the per-store model was already
   best, gain 0.01–0.02. Ten stores' worth of rows let the trees learn the
   weekly and holiday shape once and apply it where a single store's
   history is too thin.
2. **The lower-quartile weakness is gone.** The per-store XGBoost improved
   on the default screen by 0.9% or less in a quarter of store-weeks; the
   pooled one by 9.6% or less. Its worst quarter is now as good as ARIMA's.
3. **Bias is halved.** +0.25 units/day against +0.79; at the busiest store
   +1.6 against +3.5. Cross-store training regularises the level.
4. **Holiday weeks:** 0.681, the best of any method, and best at seven of
   ten stores; ARIMA 0.756, ETS 0.820.
5. **Best method per store:** pooled XGBoost at eight of ten (all but CA_4,
   where ETS leads by 0.02, and TX_1, where ARIMA leads by 0.006 over the
   per-store model and by 0.006 the pooled one leads ARIMA - a tie).

## What it means

- This is the production design and it is also the most accurate one. One
  fit per origin for the item, not ten; a new store or a new item is rows,
  not a new model. On this item it beats the best classical model by 0.026
  RMSSE and the per-store version by 0.045.
- The classical models cannot pool: ETS and ARIMA are one model per series
  by construction. The comparison that matters for a store system is
  therefore pooled XGBoost against per-series ARIMA, and pooled XGBoost
  wins on every column here except bias, where ARIMA is still centred and
  the pooled model still runs slightly high.
- The parent's headline ("the three models tie on ordinary weeks") was a
  statement about per-store XGBoost. With pooling the tie is broken.

## Cost, measured

358 pooled fits of ~119,000 rows in 972 s: 2.7 s a fit. The earlier
attempt, which fitted the same model once per store per origin, would have
taken about ten times that; it was stopped after eight hours and the
harness corrected (`src/models/xgboost_model.py`, memo per origin).

## Not done here

- Pooling wider than the item (department, category) is wired through
  `Config.pool_by` and untested.
- The pooled model's one-step residuals and its calibrated quantiles have
  not been looked at; the item-1 machinery applies unchanged.
