"""
Benchmark suites, the fixed layouts every model change is scored on.

A change only counts when it is scored on the same items, folds and metrics
as the version it replaces. A suite pins the layout. The item is chosen
separately, so one suite runs on any item.

  dev       three stores, eight weekly folds, a 20 s check while iterating
  weekly    ten stores, 52 weekly folds (the first study's layout)
  everyday  ten stores, every day an origin over the same year (the reported
            layout since item 2)

Later suites go here as well, a class-stratified item set or a new-items
suite. Adding one is adding a dict entry.
"""

from __future__ import annotations

from src.step1_problem import DEV, STUDY_ITEMS, Config

# Layout fields only. Items, pooling and the model are chosen per run.
SUITES: dict[str, dict] = {
    "dev": {"store_ids": DEV["store_ids"], "n_folds": DEV["n_folds"], "fold_step": 7},
    "weekly": {"n_folds": 52, "fold_step": 7},
    "everyday": {"n_folds": 358, "fold_step": 1},
}

# Items with a full set of cached runs today. `--item` accepts any M5 item.
KNOWN_ITEMS = {
    "fast": STUDY_ITEMS[0],  # FOODS_3_586, sold nearly every day at all stores
    "slow": "FOODS_1_021",  # declining, erratic/lumpy across stores
}


def suite_config(suite: str, item: str, pool_by: str | None = None, **overrides) -> Config:
    """The Config for one suite on one item. `item` may be a KNOWN_ITEMS key or an item id."""
    if suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}; choose from {sorted(SUITES)}")
    item_id = KNOWN_ITEMS.get(item, item)
    return Config(item_ids=(item_id,), pool_by=pool_by, **SUITES[suite], **overrides)


def suite_of(cfg: Config) -> str | None:
    """The suite whose layout matches a config, or None."""
    def same(a, b):
        return tuple(sorted(a)) == tuple(sorted(b)) if isinstance(a, tuple) else a == b

    for name, fields in SUITES.items():
        if all(same(getattr(cfg, k), v) for k, v in fields.items()) and (
            name == "dev" or not cfg.store_ids
        ):
            return name
    return None
