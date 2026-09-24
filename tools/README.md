# Tools

Scripts for running, checking and comparing the study. Each runs from the
repo root with `PYTHONPATH=.` and reads the cache and the registry; none of
them changes a forecast except `refit_all.py`, which makes new ones.

| script | what it does | when to use it |
|---|---|---|
| `refit_all.py "<note>"` | Runs every reported configuration (both items, weekly and every-day layouts, point methods and pooled variants) through `src.run`, registering each run under the note. A stage that runs past twice its estimate is killed and logged. | after any change that touches forecasts, before claiming numbers |
| `compare_runs.py "<new note>" ["<old note>"]` | For each run under the new note, finds the earlier run of the same method and config and prints the registry's paired comparison (win rate, quartiles). | right after a refit |
| `compare_forecasts.py ["<new note>" "<old note>"]` | Compares two refits forecast by forecast: share of identical forecasts, the largest difference, the dates that differ, and the RMSSE scale per series. | when a refit was expected to change nothing, or only certain days |
| `gate_identity.py` | Fits each method from scratch on the DEV and weekly layouts and compares with the last cached run. Benchmarks, ETS and ARIMA must match exactly; XGBoost differences are reported. Prints timings. | before merging a harness change |
| `recompute_check.py` | Refits per-store and pooled XGBoost on the weekly layout and compares with the cache row by row. About 6 min. | a quicker version of the gate |
| `render_items.py` | The item 3–5 figures from the cached every-day runs (RMSSE and bias by store, horizon, holiday forecasts, residuals, monthly level). Writes to `figures/items345/`. | when reviewing those items |
| `registry_figures.py` | Exports the registry (runs, current runs with shorthand, fold scores) and draws the per-run RMSSE chart, the stage-by-stage progression and the old-vs-new speed chart. Writes to `figures/registry/`. | after a refit, for a look at everything on record |
| `bench_threads.py` | Times XGBoost and ARIMA fits three ways (all cores per fit, one thread per fit, twenty single-thread fits at once) on real folds from the study. | when in doubt about the parallel harness on a new machine |

`figures/` is gitignored, like every rendered figure in the project.
