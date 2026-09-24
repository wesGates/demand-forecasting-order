# Item 6 — the run registry

Dated 2026-09-23. Infrastructure, not a model change: no forecast in the
cache was altered. Branch `item6-run-registry`.

## What was built

- `src/registry.py`: a SQLite file beside the cache
  (`cache/registry.sqlite`) with two tables, `run` (one row per cached run:
  method, suite, item, pooling, code digest, git state, time, headline
  scores against the seasonal naïve and the 28-day mean) and `fold_score`
  (one row per store-origin: RMSSE, bias, MAE, RMSE, the store's level that
  week, the kind of week). The same two tables are the planned SQL Server
  results store, so nothing here is redesigned later.
- `src/suites.py`: the fixed layouts a change is scored on — `dev`,
  `weekly`, `everyday` — chosen by name, the item chosen separately.
- `src/run.py`: `python -m src.run --suite dev --item fast --methods
  xgboost_rel --note "..."` runs (or loads) the methods, registers them, and
  compares each with its predecessor on the same layout.
- `python -m src.registry backfill | list | show | compare | last`.
- Recording wraps the harness instead of editing it, so no cache key
  changed. Rows are derived from the cache and git only; the registry is
  never edited by hand and can be rebuilt with `backfill`.

## Back-fill of the cache on disk

204 cached runs indexed in 6 min 30 s (the time is the paired scoring
against both benchmarks per run). Runs made under earlier code are kept
with `code_current = 0`. For back-filled rows `recorded_at` is the cache
file's modification time, which for re-keyed files is the re-key time, not
the run; `git_commit` is unknown. Rows recorded through `src.run` from now
on carry the true time, commit and branch.

Spot check against the findings: every-day fast mover — pooled
level-relative 0.612 (win 84.0%, median 28.0%), pooled 0.621, ARIMA 0.647,
per store 0.666, ETS 0.670; every-day declining item — moving average
0.497, ETS 0.500, pooled level-relative 0.500, pooled 0.570, per store
0.610. All match the item 3–5 findings.

## First use: is item 5's fast-mover gain real?

Paired on the 3,580 store-origins the two runs share:

```
xgboost_rel_3523527c471c  vs  xgboost_ce1fa90ffef3
  RMSSE  0.621 -> 0.612     bias  +0.25 -> +0.21
  B better in 54% of store-origins; improvement mean 0.6%,
  quartiles -6.7% / 1.2% / 9.0%
```

The level-relative pooled model is better in 54% of forecasts with a
median gain of 1.2%: a small, real-looking but marginal improvement on the
fast mover. The registry's first job was to say so. The gain on the
declining item (0.570 → 0.500) is the substantive result of item 5; the
fast-mover figure is "no cost, slight gain", which is how the item 5
findings already describe it.

## Not done

- Re-keying is not yet recorded as its own event; when the parallel harness
  (item 7) changes the cache key, the registry gains a `rekey` source and
  a note of the identity check that justified it.
- Suites for a class-stratified item set and for new items: entries in
  `src/suites.py` when those studies start.
