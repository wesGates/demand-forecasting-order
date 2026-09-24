# Item 7 — the parallel harness

Dated 2026-09-23. Infrastructure: no forecast changed, and that was
proved rather than assumed. Branch `item7-parallel-harness`, commit
`cc62970` (code) and the commit that adds this file.

## What was built

- `run_walk_forward` splits the walk-forward into tasks: one fold of one
  series when models fit per series, one fold of every series when they
  pool (a pooled model is fitted once per origin and shared, so the stores
  sit in one process). Tasks run in worker processes, forked from the
  parent so the data is shared without copying.
- One thread per fit. XGBoost runs with `n_jobs=1`; every numerical
  library is pinned to one thread at worker start, with XGBoost loaded
  under the limit. Parallelism is across fits, never inside one.
- ARIMA's order is chosen first, in parallel over series, on each series'
  first scored window, and handed to every worker, so the choice is the
  one the serial loop made.
- `n_jobs` (default: every core) changes wall time only and is not part of
  the cache key. `n_jobs=1` runs the same task function in-process.

## Why one thread per fit

The old setting gave each fit all twenty threads. A benchmark on real fits
from the study (20 XGBoost fits of ~13,000 rows; ARIMA with order search):

| setting | XGBoost, 20 fits | ARIMA, 20 fits |
|---|---|---|
| one at a time, all threads per fit (old) | 13.7 s | — |
| one at a time, one thread per fit | 4.7 s | 478 s |
| 20 at once, one thread each (new) | 1.7 s | 82 s |

A single fit of this size is too small to use twenty threads; one thread
per fit is already three times faster, and running twenty at once is eight
times faster than the old setting for XGBoost and six times for ARIMA.

## The one bug, and how it showed

The first version pinned threads in the worker *before* XGBoost was
imported. XGBoost imports lazily inside the fit, so its OpenMP pool loaded
unlimited: twenty workers with twenty busy-waiting threads each, a load
average of 200 on a 20-thread CPU, and a gate that should have taken ten
minutes still running after forty. The fix loads XGBoost under the limit
and sets the thread-limit environment variables at worker start. The
same DEV run went from crawling to 2.3 s.

## Gates, all passed before any long run

1. Tests: 107 passed (four new: per-series, pooled and ARIMA forecasts
   identical between one and three workers; the key ignores `n_jobs`).
2. Validator: 9 of 9, including the byte-identical rerun.
3. From-scratch fits with the new harness against the old cache on the
   DEV and weekly layouts, per-store and pooled, every method: identical,
   XGBoost included. This version of XGBoost's histogram method gives the
   same result whatever the thread count, so the earlier expectation of
   last-decimal differences did not materialise.

## The refit

Every reported run made again under the new harness and registered:
both items, weekly and every-day layouts, ten point methods per store and
both pooled variants. 19:55–20:35, 40 minutes wall, eight stages, none
over its estimate. Old cache files were left in place under their old
keys; the registry pairs each new run with its predecessor:

- 43 runs had an earlier counterpart: **all 43 identical**, forecast for
  forecast (win rate 0, every quartile 0.0%).
- 5 had none (benchmarks never before run on the weekly layout for the
  slow item).

Run times under the new harness (seconds; every-day = 358 origins × 10
stores):

| | ARIMA | ETS | XGBoost | level-relative | pooled XGBoost | pooled level-relative |
|---|---|---|---|---|---|---|
| weekly, fast mover | 131 | 5 | 20 | 18 | 21 | 21 |
| weekly, slow item | 102 | 7 | 7 | 7 | 7 | 6 |
| every-day, fast mover | 634 | 36 | 130 | 115 | 135 | 136 |
| every-day, slow item | 500 | 35 | 47 | 41 | 39 | 32 |

A full every-day run of every point method on one item is now about 15
minutes (was about 2 hours); the pooled models about 2 minutes each (were
16). Benchmarks take under a second.

## What it means

- Iteration is no longer the bottleneck: a DEV check is seconds, a weekly
  check three minutes, the reported layout fifteen.
- Results no longer depend on the machine's core count. Cross-OS
  identity is still not claimed (XGBoost's column subsampling can differ
  by platform).
- The registry has its first `code_current` refresh: 59 rows flipped to
  "made under earlier code" after the harness change, which is the
  intended bookkeeping.
