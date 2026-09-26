"""
Gradient-boosted trees with a Poisson objective on the raw count.

The point model minimises squared error, which treats a forecast of 2 on a
day that sold 0 the same as 12 on a day that sold 10. Sales are counts, and
on the slow item nearly half the days are zero. Poisson is the count-aware
loss the plan lists for low-volume series. Same features, same settings,
only the objective changes. It is not combined with the level-relative
target because a Poisson target cannot be negative.
"""

from __future__ import annotations

import numpy as np

from src.models.base import Context
from src.models.xgboost_model import fit_predict_xgboost


def fit_predict_xgboost_poisson(ctx: Context, **overrides) -> np.ndarray:
    """The point model's trees under a Poisson objective."""
    return fit_predict_xgboost(ctx, objective="count:poisson", **overrides)
