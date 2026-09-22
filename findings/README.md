# Findings

The notebooks are the *procedure*: they run unchanged for any item and print
what they find rather than stating it. What the findings *mean* for a
particular item is written here, one dated file per item, answering the
checklist at the end of each notebook:

- `01_explore` — is the working set clean, what is the dominant structure,
  which known-in-advance variables carry signal, does the demand class vary.
- `02_evaluate` — which methods beat the benchmarks and by how much, whether
  the ranking depends on the store and on the kind of week, how error grows
  with horizon, whether the residuals are clean, where the models lose.

These files are the raw material for the report under `report/`, which is
the public write-up. A findings file quotes numbers; the report explains them.
