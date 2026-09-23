"""
Run one or more methods on a benchmark suite and register the result.

    python -m src.run --suite dev --item fast --methods xgboost_rel --note "try X"
    python -m src.run --suite everyday --item slow --pool item_id --methods xgboost_rel

Cached methods load instantly and are still registered, so this is also the
way to put an existing run on record under a note. After each method the
run is compared with its predecessor on the same layout, if there is one.
"""

from __future__ import annotations

import argparse
import sys

from src.registry import compare, format_comparison, predecessor, record, runs
from src.suites import KNOWN_ITEMS, SUITES, suite_config


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", required=True, choices=sorted(SUITES))
    p.add_argument("--item", default="fast", help=f"{'/'.join(KNOWN_ITEMS)} or an M5 item id")
    p.add_argument("--pool", default=None, help="pool_by column, e.g. item_id")
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--note", default=None, help="what this run tries; goes in the registry")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)

    cfg = suite_config(a.suite, a.item, pool_by=a.pool)
    ids = record(cfg, a.methods, note=a.note, progress=not a.quiet)
    table = runs(where=f"run_id IN ({','.join('?' * len(ids))})", params=tuple(ids))
    cols = ["run_id", "suite", "item_ids", "pool_by", "rmsse_mean", "bias_mean", "win_vs_ref", "impr_median", "run_seconds"]
    print(table[cols].to_string(index=False))
    for run_id in ids:
        prev = predecessor(run_id)
        if prev:
            print(format_comparison(compare(prev, run_id)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
