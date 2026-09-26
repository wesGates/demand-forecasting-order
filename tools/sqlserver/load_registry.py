"""
Load the run registry into SQL Server and run a few queries against it.

The registry is a SQLite file beside the cache (`cache/registry.sqlite`)
with two tables, `run` (one row per cached run) and `fold_score` (one row
per store-origin). This script creates the same two tables in a SQL Server
database, loads them, and prints three queries a planner or a reviewer
would ask. It is the results store the plan describes, in its simplest
form. Run from the repository root:

    PYTHONPATH=. python tools/sqlserver/load_registry.py

Connection comes from the environment, with the defaults the compose file
uses: MSSQL_HOST (localhost), MSSQL_PORT (1433), MSSQL_USER (sa),
MSSQL_PASSWORD (required), MSSQL_DATABASE (forecasting).
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pymssql

REGISTRY = Path("cache/registry.sqlite")

# Column types for SQL Server. The SQLite file stores everything loosely; here
# the keys are fixed-width strings and the numbers get their proper types.
RUN_COLUMNS = {
    "run_id": "VARCHAR(64) NOT NULL PRIMARY KEY", "method": "VARCHAR(40) NOT NULL", "kind": "VARCHAR(16)",
    "suite": "VARCHAR(16)", "item_ids": "VARCHAR(200)", "store_ids": "VARCHAR(200)", "pool_by": "VARCHAR(16)",
    "fold_step": "INT", "n_folds": "INT", "horizon": "INT", "seed": "INT", "config_json": "NVARCHAR(MAX)",
    "code_digest": "CHAR(40)", "code_current": "BIT", "git_branch": "VARCHAR(80)", "git_commit": "CHAR(40)",
    "git_dirty": "BIT", "recorded_at": "DATETIME2", "source": "VARCHAR(16)", "run_seconds": "FLOAT",
    "note": "NVARCHAR(400)", "n_scored": "INT", "rmsse_mean": "FLOAT", "rmsse_median": "FLOAT",
    "rmsse_normal": "FLOAT", "rmsse_holiday": "FLOAT", "bias_mean": "FLOAT", "win_vs_ref": "FLOAT",
    "impr_mean": "FLOAT", "impr_q1": "FLOAT", "impr_median": "FLOAT", "impr_q3": "FLOAT",
    "win_vs_stress": "FLOAT", "impr_median_stress": "FLOAT", "cache_file": "VARCHAR(200)",
    "n_unscored": "INT", "n_fallback": "INT",
}
FOLD_COLUMNS = {
    "run_id": "VARCHAR(64) NOT NULL", "id": "VARCHAR(40) NOT NULL", "item_id": "VARCHAR(20)",
    "store_id": "VARCHAR(8)", "fold": "INT", "origin": "DATE NOT NULL", "week_kind": "VARCHAR(8)",
    "rmsse": "FLOAT", "bias": "FLOAT", "mae": "FLOAT", "rmse": "FLOAT", "mean_actual": "FLOAT",
    "n_days": "INT", "unscored": "BIT", "fallback": "BIT",
}

QUERIES = {
    "best current method per item, every-day layout": """
        SELECT item_ids, method, pool_by, rmsse_mean
        FROM (
            SELECT item_ids, method, pool_by, rmsse_mean,
                   ROW_NUMBER() OVER (PARTITION BY item_ids ORDER BY rmsse_mean) AS rank_in_item
            FROM run WHERE code_current = 1 AND suite = 'everyday'
        ) ranked
        WHERE rank_in_item <= 3
        ORDER BY item_ids, rmsse_mean""",
    "the final model by store on the fast mover": """
        SELECT f.store_id, ROUND(AVG(f.rmsse), 3) AS rmsse, ROUND(AVG(f.bias), 2) AS bias, COUNT(*) AS weeks
        FROM fold_score f JOIN run r ON r.run_id = f.run_id
        WHERE r.code_current = 1 AND r.suite = 'everyday' AND r.item_ids = 'FOODS_3_586'
          AND r.method = 'xgboost_rel' AND r.pool_by = 'item_id'
        GROUP BY f.store_id ORDER BY rmsse""",
    "holiday weeks against normal weeks, per method, fast mover": """
        SELECT r.method, r.pool_by,
               ROUND(AVG(CASE WHEN f.week_kind = 'normal' THEN f.rmsse END), 3) AS normal_weeks,
               ROUND(AVG(CASE WHEN f.week_kind = 'holiday' THEN f.rmsse END), 3) AS holiday_weeks
        FROM fold_score f JOIN run r ON r.run_id = f.run_id
        WHERE r.code_current = 1 AND r.suite = 'everyday' AND r.item_ids = 'FOODS_3_586'
        GROUP BY r.method, r.pool_by ORDER BY normal_weeks""",
}


def connect(database: str | None):
    password = os.environ.get("MSSQL_PASSWORD")
    if not password:
        sys.exit("set MSSQL_PASSWORD (the SA password the container was started with)")
    return pymssql.connect(
        server=os.environ.get("MSSQL_HOST", "localhost"), port=int(os.environ.get("MSSQL_PORT", "1433")),
        user=os.environ.get("MSSQL_USER", "sa"), password=password, database=database or "master", autocommit=True,
    )


def create_database(name: str) -> None:
    with connect(None) as con:
        cur = con.cursor()
        cur.execute(f"IF DB_ID('{name}') IS NULL CREATE DATABASE [{name}]")


def create_tables(con) -> None:
    cur = con.cursor()
    cur.execute("IF OBJECT_ID('fold_score') IS NOT NULL DROP TABLE fold_score")
    cur.execute("IF OBJECT_ID('run') IS NOT NULL DROP TABLE run")
    cur.execute("CREATE TABLE run (" + ", ".join(f"[{c}] {t}" for c, t in RUN_COLUMNS.items()) + ")")
    cur.execute(
        "CREATE TABLE fold_score (" + ", ".join(f"[{c}] {t}" for c, t in FOLD_COLUMNS.items())
        + ", PRIMARY KEY (run_id, id, origin), FOREIGN KEY (run_id) REFERENCES run(run_id))"
    )


def load_table(con, name: str, frame: pd.DataFrame, columns: dict[str, str], rows_per_statement: int = 500) -> int:
    """
    Insert many rows per statement. pymssql has no bulk copy, and its
    executemany sends one statement per row, about 400 rows a second on
    the 722k score rows. A multi-row VALUES list is more than ten times
    faster. SQL Server caps a statement at 1,000 rows and 2,100 bound
    parameters, and 500 rows of 15 columns would exceed the parameter cap,
    so the values are quoted into the statement text with pymssql's own
    parameter substitution rather than bound.
    """
    from pymssql import _mssql  # the quoting pymssql uses for parameters

    cols = list(columns)
    frame = frame[cols].astype(object).where(frame[cols].notna(), None)
    head = f"INSERT INTO {name} (" + ", ".join(f"[{c}]" for c in cols) + ") VALUES "
    placeholders = "(" + ", ".join(["%s"] * len(cols)) + ")"
    cur = con.cursor()
    rows = [tuple(r) for r in frame.itertuples(index=False, name=None)]
    for start in range(0, len(rows), rows_per_statement):
        block = rows[start:start + rows_per_statement]
        values = ", ".join(_mssql.substitute_params(placeholders, r).decode() for r in block)
        cur.execute(head + values)
    return len(rows)


def main() -> int:
    database = os.environ.get("MSSQL_DATABASE", "forecasting")
    src = sqlite3.connect(REGISTRY)
    run = pd.read_sql_query("SELECT * FROM run", src)
    fold = pd.read_sql_query("SELECT * FROM fold_score", src)
    # SQLite stores booleans as 0/1 integers and dates as text; SQL Server wants real types.
    for c in ("code_current", "git_dirty"):
        run[c] = run[c].astype("Int64")
    for c in ("unscored", "fallback"):
        fold[c] = fold[c].astype("Int64")
    run["recorded_at"] = pd.to_datetime(run["recorded_at"], utc=True).dt.tz_localize(None)
    fold["origin"] = pd.to_datetime(fold["origin"]).dt.date

    create_database(database)
    with connect(database) as con:
        create_tables(con)
        n_run = load_table(con, "run", run, RUN_COLUMNS)
        n_fold = load_table(con, "fold_score", fold, FOLD_COLUMNS)
        print(f"loaded {n_run} runs and {n_fold} store-origin scores into {database}")
        cur = con.cursor(as_dict=True)
        for title, sql in QUERIES.items():
            cur.execute(sql)
            print(f"\n--- {title}")
            print(pd.DataFrame(cur.fetchall()).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
