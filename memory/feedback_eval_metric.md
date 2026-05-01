---
name: ablation eval uses δ + AUC@30 averaged
description: Judge module usefulness by averaging δ (mean of δ@0.5/1/2m) AND AUC@30 — not just δ alone
type: feedback
originSessionId: 2593d9fb-88bf-4bdb-9677-2c7041f59fad
---
When evaluating whether an ablation module is useful, look at BOTH the delta-mean metric AND AUC@30. Average them together — only if both improve does the module count as beneficial.

**Why:** δ alone can be noisy or move in opposite direction from AUC. The user wants robust evidence that a module truly helps.

**How to apply:** When comparing ADD-N vs previous best in forward-add (or any ablation comparison), report both `Mean(δ@0.5,1,2)` and `AUC@30`, then their average. Decision rule (>0.10 improvement) applies to the combined average, not just δ.
