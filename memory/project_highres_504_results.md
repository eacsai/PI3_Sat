---
name: highres @504 evaluation results (C1 / C2 / Pi3_ori)
description: Final clean @504 numbers for stage-2 dynamic high-res models (C1_highres, C2_highres, Pi3_ori_highres), recomputed from per-sample CSVs
type: project
originSessionId: 2593d9fb-88bf-4bdb-9677-2c7041f59fad
---
Stage-2 dynamic high-res evaluation on **googlestreet_sat** test split (N=100 sequences, 504×504 input).

All numbers below are **mean across 100 samples** computed from
`outputs/<run>/googlestreet_sat/_all_samples.csv` (not from the aggregate
`*-<suffix>.csv`, which can carry stale rows from previous runs — see
`feedback_csv_dirty_aggregates.md`).

| Metric | C1_highres | C2_highres | Pi3_ori_highres |
|---|---|---|---|
| δ@0.5 / 1 / 2 | 43.42 / 64.16 / 83.46 | 40.63 / 61.22 / 81.44 | 38.37 / 59.24 / 80.99 |
| **δ-Mean** | **63.68** 🥇 | 61.10 | 59.53 |
| **AUC@30** | **81.83** 🥇 | 79.98 | 80.62 |
| Acc / Comp | **1.12 / 1.28** 🥇 | 1.21 / 1.37 | 1.22 / 1.47 |
| PCK@2m UAV / Grd | **85.0 / 43.1** 🥇 | 82.9 / 42.7 | 10.9 / 15.2 |
| PCK@5m UAV / Grd | **96.0 / 84.7** 🥇 | 93.9 / 83.2 | 37.6 / 50.3 |
| Train best val | (best at ep 28+) | 0.0641 (ep 28) | 0.0692 (ep 13) |

**Why:** confirms C1_highres remains best across δ, AUC, Acc/Comp, and PCK
under the bug-fixed eval (proj formula uses `width/data_w` and
`height/data_h` correctly for non-square sat). C2 (camera-side
sat-injection) lowered val loss but tested *worse* than C1 — same pattern as
the lowres @224 finding (camera-branch sat injection adds noise without net
gain). PCK numbers show the sat_mpp_head advantage is dramatic for C1/C2 vs
Pi3_ori (~85% vs 11% UAV @2m).

**How to apply:** these are the canonical highres @504 baselines. When
adding new highres variants, compare δ-Mean / AUC@30 / PCK against this
table. Re-run `python -c 'import pandas as pd; ...'` over per-sample CSV to
verify aggregate CSV isn't stale.

## Eval scripts used
- C1_highres / C2_highres → `eval_sat.py` (uses model's `sat_mpp`)
- Pi3_ori_highres → `eval.py` (uses `recover_focal_shift` since vanilla Pi3 has no sat_mpp_head)
- Both call `KittiPairDataset`-equivalent `googlestreet_sat` config at 504×504.

## Bug fixes applied this session (verify present)
1. `eval.py:284-285` — sat-projection uses `width.item()` for u and
   `height.item()` for v (was hardcoded `224`); supports non-square sat.
2. `eval_sat.py:308` — same fix in the no-mpp fallback path.
3. `base_trainer_accelerate.py:log_all` — replaced removed
   `Image.isImageType` with `isinstance(v, Image.Image)` and added 0-D
   tensor scalar logging.
