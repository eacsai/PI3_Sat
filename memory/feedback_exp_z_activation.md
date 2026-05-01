---
name: all sat experiments use z = torch.exp(z)
description: All Pi3 sat experiments must use z = torch.exp(z) for depth activation, never F.softplus(z) — matches original Pi3
type: feedback
originSessionId: 2593d9fb-88bf-4bdb-9677-2c7041f59fad
---
All future sat ablation experiments (ADD-N, etc.) must use `z = torch.exp(z)` for the depth activation in `pi3_training_sat.py:460`, matching the original Pi3 (`pi3_training.py:281`). Never use `F.softplus(z) + 1e-6`.

**Why:** softplus produces a different depth distribution than exp, breaking comparability with Pi3_ori baseline (Mean=57.20). Exp14 (exp) > Exp15 (softplus) confirms exp is better. The user wants apples-to-apples comparison across all ablations.

**How to apply:** Before launching any new experiment based on `pi3_training_sat.py`, verify line 460 reads `z = torch.exp(z)`. If a previous experiment snapshot used softplus, results from it are invalid for comparison.
