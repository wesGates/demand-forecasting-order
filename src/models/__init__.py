"""
The forecasters, one module each, so that a change to one of them invalidates
only its own cached predictions.

  base.py           Context (what a forecaster may see) and shared helpers
  benchmarks.py     the six simple methods (FPP Ch. 5)
  xgboost_model.py  gradient-boosted trees on the supervised matrix
  ets_model.py      Holt-Winters exponential smoothing (FPP Ch. 8)
  arima_model.py    seasonal ARIMA with calendar regressors (FPP Ch. 9-10)

`step4_models.py` is the registry that maps names to these, and the only
module the rest of the project imports from.
"""
