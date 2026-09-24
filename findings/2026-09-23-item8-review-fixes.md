# Item 8 — the review fixes, and what they changed

Dated 2026-09-23 (night). Branch `item8-review-fixes`. A code review found
one leak, two registry bugs, two edge cases and a set of risks; all are
fixed and tested (114 tests, validator 10/10 with a new harness-parity
check). This file records the one fix that changes numbers, the refit under
it, and the corrected headline tables. The full list is in `GLOSSARY.md`
("Rules added by the review") and the plan's status table.

## The leak

`impute_closures` filled a closure day with the mean of the same weekday
over the four weeks before **and after** it. Christmas 2015 lies inside the
scored year, so its imputed value carried one-eighth each of Jan 1, 8, 15
and 22, 2016. That value then entered the lags, the rolling means, the
training rows and the seasonal naïve of every fold with an origin from
Dec 25 to Jan 21, and the four earlier Christmases entered every training
history and the RMSSE scale the same way. Fix: the four preceding weeks
only, skipping days inside a holiday window (FPP §5.10 for the rule; §13.7
for closure-as-missing). `CACHE_VERSION` 3. On the fast mover 50 rows
changed, on the slow item 29 - all closure days. The busiest store's
Christmas 2015 moved from 100.0 (blended with January) to 119.3 (December
Fridays).

## The refit

Every reported run made again under the corrected data (registry note
"item 8 refit"), 21:49–22:33, eight stages, 44 minutes, none over its
estimate. No fold was unscored and no method fell back on any fold
(`n_unscored` = `n_fallback` = 0 across all 48 runs).

### Forecast by forecast, against the item 7 runs

| method | forecasts identical | where they differ |
|---|---|---|
| naïve, drift, seasonal naïve, lag-364 naïve | 99.7% | only the days that read Dec 25, 2015: Dec 26 – Jan 1 (naïve, drift), Jan 1 (seasonal naïve), Dec 24 (lag 364) |
| 28-day moving average | 92% | the windows containing Dec 25: Dec 26 – Jan 28 |
| mean | 0% (max difference 0.03 units) | every forecast, by the four earlier Christmases in the history |
| ETS, ARIMA, XGBoost (all variants) | 0–6% | every forecast, by small amounts: their training histories changed |

The RMSSE scale per series moved by at most 0.14%, so every RMSSE shifts by
that factor even where forecasts are identical.

### Run by run (every-day layout, paired on 3,580 store-origins)

| method | item 7 → item 8 RMSSE | paired median change |
|---|---|---|
| fast mover, level-relative pooled | 0.6119 → 0.6111 | +0.04% |
| fast mover, pooled | 0.6212 → 0.6218 | −0.08% |
| fast mover, ARIMA | 0.6472 → 0.6476 | +0.05% |
| fast mover, per-store XGBoost | 0.6664 → 0.6646 | +0.19% |
| fast mover, ETS | 0.6700 → 0.6704 | +0.04% |
| slow item, level-relative pooled | 0.5004 → 0.5006 | +0.01% |
| slow item, pooled | 0.5701 → 0.5701 | +0.05% |
| slow item, moving average | 0.4971 → 0.4973 | +0.01% |
| slow item, ETS | 0.4998 → 0.4998 | +0.01% |

Third-decimal changes; no ranking moves.

## Corrected headline tables (every-day layout)

Fast mover (`FOODS_3_586`):

| method | RMSSE all | normal | holiday | bias | win vs SN7 | median impr. | Q1 |
|---|---|---|---|---|---|---|---|
| XGBoost, pooled, level-relative | **0.611** | 0.599 | 0.665 | +0.21 | 84.4% | 27.8% | 11.1% |
| XGBoost, pooled | 0.622 | 0.608 | 0.683 | +0.26 | 83.5% | 26.9% | 9.0% |
| XGBoost, per store, level-relative | 0.639 | 0.623 | 0.709 | +0.25 | 81.1% | 25.2% | 7.1% |
| ARIMA | 0.648 | 0.622 | 0.758 | +0.00 | 83.0% | 25.4% | 9.3% |
| XGBoost, per store | 0.665 | 0.647 | 0.741 | +0.76 | 76.1% | 23.3% | 1.5% |
| ETS | 0.670 | 0.635 | 0.822 | −0.06 | 83.6% | 21.7% | 7.4% |
| 28-day moving average | 0.795 | 0.771 | 0.901 | −0.03 | 59.4% | 8.7% | −17.7% |
| seasonal naïve | 0.864 | 0.825 | 1.035 | −0.04 | | | |

Best method by store: level-relative pooled at seven, pooled at three.
Pooling improves nine of ten stores over the per-store model (WI_1 is
0.002 worse); the quietest, CA_4, 0.707 → 0.586.

Slow, declining item (`FOODS_1_021`):

| method | RMSSE all | normal | holiday | bias | win vs SN7 |
|---|---|---|---|---|---|
| 28-day moving average | **0.497** | 0.497 | 0.500 | +0.08 | 82.1% |
| ETS | 0.500 | 0.499 | 0.504 | +0.14 | 81.1% |
| XGBoost, pooled, level-relative | 0.501 | 0.499 | 0.508 | +0.18 | 81.9% |
| XGBoost, per store, level-relative | 0.507 | 0.505 | 0.513 | +0.19 | 79.8% |
| ARIMA | 0.519 | 0.512 | 0.550 | +0.33 | 77.0% |
| XGBoost, pooled | 0.570 | 0.565 | 0.591 | +0.57 | 69.7% |
| XGBoost, per store | 0.610 | 0.605 | 0.631 | +0.95 | 62.5% |
| seasonal naïve | 0.687 | 0.691 | 0.671 | +0.03 | |

Best method by store: moving average at five, ETS at three, level-relative
pooled at two.

## Earlier findings files

Items 2–5 keep their numbers as the record of each stage; every number in
them that appears in a later summary is taken from these tables instead.
The first-study report (weekly layout, the parent repository) was made on
the old imputation; its numbers move in the third decimal under the
correction.
