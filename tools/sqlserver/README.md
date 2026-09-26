# The run registry in SQL Server

The registry (`cache/registry.sqlite`, tables `run` and `fold_score`) is
the record of every run: its settings, the code it ran under, the commit,
the headline scores, and one row per store-origin. This folder puts the
same two tables in SQL Server, which is what a shared results store would
be, and queries them.

1. Start the server (Docker, SQL Server 2022 Developer edition):

       MSSQL_PASSWORD='<a strong password>' docker compose -f tools/sqlserver/docker-compose.yml up -d

2. Load the registry and run the example queries:

       MSSQL_PASSWORD='<the same password>' PYTHONPATH=. python tools/sqlserver/load_registry.py

   It creates database `forecasting`, drops and recreates the two tables
   (`run` keyed by `run_id`; `fold_score` keyed by run, series and origin,
   with a foreign key to `run`), loads them, and prints three queries: the
   best current methods per item, the final model by store, and holiday
   against normal weeks per method.

The SQLite file stays the source of truth. The load is a copy, repeatable
at any time. `pymssql` is the driver (no ODBC install needed).

The port is bound to localhost only, the password lives in the environment
and never in a file under git, and `sa` is for this setup step. A shared
store would get its own login with rights on the `forecasting` database
alone, read-only for anything that just queries.
