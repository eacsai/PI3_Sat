---
name: aggregate eval CSVs append, never overwrite — use per-sample CSV
description: When reading eval results, recompute aggregates from per-sample _all_samples.csv; the aggregate `*-<suffix>.csv` accumulates rows from every run including buggy ones
type: feedback
originSessionId: 2593d9fb-88bf-4bdb-9677-2c7041f59fad
---
Eval scripts (`mv_recon/eval.py`, `eval_sat.py`, `eval_kitti_sat.py`) write
results in two places per dataset:

- **Per-sample CSV** (`outputs/<run>/<dataset>/_all_samples.csv`)
  → **overwritten** each run; one row per sequence; this is the truth
- **Aggregate CSV** (`outputs/<run>/<dataset>-<suffix>.csv`)
  → **appended** every run; multiple data rows accumulate over time, even
  rows from runs with bugs (hardcoded 224, wrong axis fix, etc.)

**Why:** I burned a turn this session reporting C1_highres @504 numbers
from the *first* aggregate CSV row (an old buggy run: δ@2m=76.7, AUC=79.5)
when the current truth was the *second* row matching per-sample mean
(δ@2m=83.5, AUC=81.8). User caught it by checking the per-sample CSV
themselves.

**How to apply:**
- For any "what's the result of run X" question, prefer the per-sample CSV
  + recompute mean/median in pandas. Don't trust the aggregate CSV's row
  count — it can carry stale data from previous runs.
- When deleting/restarting a run, also delete the aggregate CSV (or the
  whole output dir) so it doesn't accumulate. `rm -rf outputs/<run>` is
  safer than relying on overwrite semantics.
- Validation snippet:
  ```python
  import pandas as pd
  df = pd.read_csv('outputs/<run>/<dataset>/_all_samples.csv')
  print(df[['δ@0.5m','δ@1m','δ@2m','AUC@30','Acc-mean','Comp-mean']].mean())
  ```
