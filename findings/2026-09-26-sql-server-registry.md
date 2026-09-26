# The run registry in SQL Server

Dated 2026-09-26. Infrastructure, no forecast changed. The registry's two
tables (`run`, one row per cached run, and `fold_score`, one row per
store-origin) were loaded into SQL Server 2022 running in Docker, and
queried there. This is the results store the plan describes, in its
simplest form: the same tables, the same keys, a foreign key between them.

## What was built

- `tools/sqlserver/docker-compose.yml`: SQL Server 2022 Developer edition,
  port bound to localhost, data in a named volume.
- `tools/sqlserver/load_registry.py`: creates database `forecasting` and
  the two tables with proper types (`run_id` as the primary key of `run`,
  and `fold_score` keyed by run, series and origin, with a foreign key to
  `run`), loads
  them from `cache/registry.sqlite`, and prints three queries. `pymssql`
  as the driver, no ODBC install. The password comes from the environment.
- The load: 381 runs and 722,088 store-origin scores in 2 min 52 s, with
  500-row INSERT statements. The first version used `executemany`, one
  statement per row, and ran at about 400 rows a second, which is 30
  minutes for the same load.

## The queries, as printed

Best current methods per item on the every-day layout (top three):

| item | method | pool | RMSSE |
|---|---|---|---|
| FOODS_1_021 | moving_average_28 | | 0.497 |
| FOODS_1_021 | ets | | 0.500 |
| FOODS_1_021 | xgboost_rel | item_id | 0.501 |
| FOODS_3_586 | xgboost_rel | item_id | 0.611 |
| FOODS_3_586 | xgboost | item_id | 0.622 |
| FOODS_3_586 | xgboost_rel | | 0.639 |

The final model by store on the fast mover: WI_3 0.481, TX_2 0.526, CA_4
0.562, CA_1 0.566, TX_1 0.588, TX_3 0.619, WI_1 0.631, WI_2 0.660, CA_3
0.673, CA_2 0.803, 358 weeks each. Holiday against normal weeks per
method: the final model 0.599 / 0.665, ARIMA 0.622 / 0.758, ARIMA without
inputs 0.628 / 0.797, ETS 0.635 / 0.822, ARMA 0.668 / 0.817. Every number
matches the item-8 findings and the report, which is the point of the
check: the SQL copy reproduces the SQLite record.

## What it says

1. The registry's design transfers as is. Nothing was redesigned for SQL
   Server. The two-table schema and the keys are the ones the SQLite file
   has had since item 6.
2. The queries a reviewer asks (best method, per-store breakdown, holiday
   split) are one SELECT each, with the `run` table's `code_current` flag
   doing the filtering.
3. The SQLite file stays the source of truth and the load is repeatable
   (it drops and recreates the tables). A shared store would run the load
   after each refit, or write runs to both.

## Not done

- A login with rights on the `forecasting` database only, read-only for
  querying. The load used `sa`, which is a setup credential.
- Writing runs to SQL Server directly from `src.run`.
- Per-forecast rows (the cached predictions themselves), which the plan's
  design includes and this load leaves in the parquet cache.
