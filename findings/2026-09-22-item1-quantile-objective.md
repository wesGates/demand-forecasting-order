# Item 1 — XGBoost on the quantile objective, against calibrated quantiles

Dated 2026-09-22. Step-7 layout (52 folds, Sundays), `FOODS_3_586`, ten
stores. Fitted quantiles come from one multi-level model per store and
origin (`reg:quantileerror`, τ = 0.3, 0.5, 0.7, 0.9); calibrated quantiles
are a point forecast plus the per-horizon τ-quantile of that method's own
daily errors over the prior year (May 2014 – May 2015). Both scored on the
same 3,640 scored store-days per horizon. Relative pinball = pinball ÷ the
store's mean daily sales; lower is better; compare within a column.

## Daily quantiles

| method | source | τ 0.3 | τ 0.5 | τ 0.7 | τ 0.9 |
|---|---|---|---|---|---|
| ARIMA | calibrated | **0.175** | **0.201** | **0.180** | **0.099** |
| ETS | calibrated | 0.183 | 0.209 | 0.186 | 0.103 |
| moving average (28) | calibrated | 0.181 | 0.209 | 0.188 | 0.104 |
| XGBoost | fitted (quantile objective) | 0.185 | 0.211 | 0.187 | 0.102 |
| XGBoost | calibrated (point + error quantile) | 0.185 | 0.212 | 0.189 | 0.102 |
| seasonal naïve | calibrated | 0.233 | 0.264 | 0.233 | 0.125 |

Coverage (share of days with actual ≤ q; target = τ):

| method | source | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|---|
| ARIMA | calibrated | 0.29 | 0.49 | 0.69 | 0.89 |
| XGBoost | calibrated | 0.30 | 0.50 | 0.70 | 0.91 |
| XGBoost | fitted | 0.38 | 0.54 | 0.68 | 0.87 |

Holiday weeks only: ARIMA 0.226 at τ 0.5, fitted XGBoost 0.227, calibrated
XGBoost 0.228, ETS 0.242, moving average 0.243.

## Weekly totals from the fitted model

Summing the fitted daily τ-quantiles over the week gives an order level
that over-covers above 0.5 and under-covers below it, as a sum of quantiles
must: coverage 0.28 / 0.52 / 0.79 / 0.97 at τ 0.3 / 0.5 / 0.7 / 0.9, and
relative weekly pinball 0.111 / 0.119 / 0.105 / 0.064 against 0.114 /
0.116 / 0.104 / 0.052 for the calibrated point model. The weekly order
still needs the calibration step.

## What it says

1. Training on the quantile objective did not beat calibrating the point
   model after the fact. The two XGBoost rows are within 0.002 at every τ.
   On this item the point model's errors, once measured on a prior year,
   carry the same information the quantile objective learns directly.
2. The fitted model's tails are slightly too narrow: 0.38 covered at τ 0.3
   and 0.87 at τ 0.9, against 0.30 and 0.91 for the calibrated version.
   Pinball trained on 12,000 rows does not pin the tails as well as an
   empirical quantile of 364 held-out errors does.
3. ARIMA with calibrated quantiles is best at every τ, daily as well as
   weekly, ordinary and holiday weeks alike. The finding from the parent's
   prototype survives at the daily level.
4. The median-quantile model scores 0.677 RMSSE against the point model's
   0.665 and carries half its bias (+0.44 against +0.88 units/day).

## Cost

One quantile fit costs about 26 point fits: pinball loss converges more
slowly, so early stopping keeps 178 trees against 51, and four levels grow
four trees per round. Measured idle, one fit: point 0.1 s, four-level
quantile 2.6 s, single-level 0.9 s. The 52-fold run took 87 minutes
because two test suites shared the machine (load average ≈ 20 on 20
cores); idle it would take about 23. On the every-day layout this is
roughly 2.5 hours for the quantile model alone.

## Consequence for the plan

The quantile objective stays in the registry as a comparator but is not
the route to a better order on this item. The order comes from the best
point forecaster plus calibration, which today is ARIMA. Item 2 (every day
an origin) and item 3 (pooling) proceed with the point models; the quantile
model is re-run only where its comparator role is needed.
