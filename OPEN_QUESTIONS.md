# Open questions and parked decisions

A living list. Each entry says what was found, what the evidence is, what the
options are, and what would settle it. Resolved items move to the bottom with
the decision recorded, so the reasoning is not lost.

Last updated: 2026-09-21.

---

## Open

### 1. XGBoost over-forecasts at the busiest store — parked

**Found:** residual diagnostics (FPP §5.4) for XGBoost at TX_2 showed a mean
residual of **+7.66 units/day** on a ~100/day store — a systematic ~7.5%
over-forecast — with a right-skewed tail to +40, on the original 8 spring
folds. ETS at the same store was essentially unbiased (+0.80).

**After the 52-fold rerun (2026-09-21):** smaller but still there. XGBoost
bias at TX_2 is **+3.9 units/day** (~4%), and the over-forecast is no longer
one store's: CA_4 +2.7, TX_3 +2.7, CA_1 +2.6, while CA_2 runs −3.6. ARIMA
and ETS stay within ±0.5 at nine of ten stores. Across all stores XGBoost's
mean bias is +0.9 against −0.1 for both classical models.

**What passes:** the residuals are uncorrelated (Ljung-Box p = 0.50), so the
method has taken everything *predictable*; the failure is in the level, not
the pattern.

**Options:**
- FPP's remedy: *"if the residuals have mean m, add m to all forecasts."* A
  per-store bias correction estimated on training folds.
- Investigate cause first: the held-out weeks are the low part of the annual
  cycle, and TX_2 was much higher in 2011–12. The rolling features should
  track the level, but the model may be leaning on `day_of_year`.
- Leave it, and report it as a finding.

**Parked until** the 52-fold rerun (2026-09-21) is read: a full year of
scored weeks may change the picture, and the holiday-proximity features give
the model a way to attribute spikes to the calendar instead of the level.

### 2. Horizon is confounded with weekday

**Found:** every fold origin is a Sunday, because origins step by exactly 7
days. So h = 1 is always Monday and h = 7 is always Sunday. The
error-by-horizon plot (FPP §5.10, figure 5.24) therefore mixes "how far
ahead" with "which weekday" — and Sunday is the highest-volume,
highest-variance day. That is why seasonal naive's error at h = 1 is *lower*
than at h = 7, and why the curves are not the textbook monotone rise.

**Why the book's example does not have this:** it steps origins by one day,
so they land on every weekday.

**Options:**
- Report the confound and read the horizon plot with it in mind (current).
- Add a `fold_step` field so origins can step by a number coprime to 7
  (e.g. 8 with 7-day windows: origins rotate through the week, with one-day
  gaps between scored windows).
- Step by 1 as FPP does — ~1,500 origins per store, overlapping windows,
  about two hours of compute.

**Decides it:** whether the horizon plot is going to carry weight in the
write-up. If it is, fix it; if RMSSE-by-store is the headline, note it.

---

## Next steps, not yet started

- **Pooled run.** `Config(pool_by="item_id")` is wired and untested on real
  data. One config flip, ~1 minute. Adds the cross-learning row to every
  table.
- **% improvement quartiles.** Per-series improvement over a benchmark,
  reported as Q1 / median / Q3 alongside win rate. ~10 lines in `summarise`.
- **Two calendar features.** Days until the next holiday, and days until the
  next SNAP day — both known years ahead, both cheap. Currently only on/off
  flags exist.
- **`min_train_days`.** Currently 365. Two full years (730) is a common
  choice and would cost nothing on the current item.
- **ARIMA.** ETS is the classical contender now; ARIMA (FPP Ch. 9) would be
  one registry entry if a second is wanted.
- **The write-up.** A dated report under `findings/` for the current item,
  then the final `.ipynb`.

---

## Resolved

*(decisions recorded here as they are made)*

- **Fold count: 52.** FPP prescribes none (§5.10's example uses every
  origin); the only sizing text is §5.8's "about 20% of the sample, at least
  as long as the horizon". One year of weekly folds is ~20% of the history
  and is the shortest layout that scores every season once. Every table is
  reported split into *normal* and *holiday* folds so the two are never
  averaged together. Decided 2026-09-21.
- **Which holidays count.** Measured, not assumed: `step3_explore.event_effects`
  gives each calendar event's sales ratio to a same-weekday baseline. Seven
  events clear a 15% bar on the day or its two-day run-up (Christmas,
  Thanksgiving, Labor Day, Independence Day, Valentine's Day, New Year,
  Easter). They drive `is_holiday` / `days_to_holiday` / `days_since_holiday`
  and the holiday-fold split. Decided 2026-09-21.
- **Christmas Day is a closure, not demand.** Every M5 store records zero;
  treated as missing per FPP §13.7 - imputed with the same-weekday mean of
  the surrounding weeks so the following week's features are sane, flagged
  `closure`, and excluded from every score. Decided 2026-09-21.
- **ARIMA joins the comparison, ARMA does not.** Seasonal ARIMA with holiday,
  pre-holiday and SNAP regressors (FPP Ch. 9-10); order chosen once per series
  by AICc at fixed differencing (§9.7). A plain ARMA is ARIMA without the
  seasonal term and cannot hold a weekly pattern on daily data. Decided
  2026-09-21.
- **Lag-364 seasonal naive added.** "This day last year" is what an orderer
  looks at before a holiday; the lag-7 version is the default screen. Both
  are benchmarks so the holiday-week comparison is against what a good
  orderer does, not only the default. Decided 2026-09-21.

- **Private filename in `34eddc0`'s `.gitignore`** — left as is. It is a
  filename, not content; the pattern was replaced with a wildcard from the
  next commit onward so it does not recur. Decided 2026-09-17.
- **RMSSE scaling** — lag 7 (seasonal naive, FPP §5.8's rule for seasonal
  data), denominator computed once per series over all pre-holdout training
  data. Alternatives (lag 1 for M5 comparability; per-fold denominator) are
  one-line switches on `Config`. Decided 2026-09-12.
- **Price as a feature** — off, via `Config.use_price`. For the current item
  price takes three values in five years on the same two dates at every
  store: a clock, not a variable. Decided 2026-09-11. **Checked
  2026-09-21** on the 52-fold layout, XGBoost only: price on gives mean
  RMSSE 0.674 against 0.665 off (normal 0.661 vs 0.653, holiday 0.728 vs
  0.717), bias +0.53 vs +0.89. A hair worse on accuracy, a little less
  over-forecast, within noise either way - the flag stays off and the
  report can quote the comparison.
- **Availability screen** — measure and warn (`max_zero_run > 30`), do not
  silently trim. Decided 2026-09-10.
- **One item across ten stores** — scope reduced from three items to one for
  time. Consequence: every store is `smooth`, so the study answers "which
  method wins on a fast mover, and does it depend on the store" rather than
  "does the best method change with demand class." Decided 2026-09-11.
