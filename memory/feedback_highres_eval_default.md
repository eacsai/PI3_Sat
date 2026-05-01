---
name: 504 eval defaults to highres weights
description: When user asks to test "best weights @504", they mean highres-trained best, not lowres baseline weights
type: feedback
originSessionId: 2593d9fb-88bf-4bdb-9677-2c7041f59fad
---
When user requests testing "最好的权重" / "best weights" at 504 resolution, they mean the **highres-trained** best_model checkpoints (e.g. `outputs/saved_runs/{model}_highres/ckpts/best_model`), NOT the lowres baseline checkpoints (e.g. `outputs/saved_runs/C1/ckpts/best_model`).

**Why:** Lowres weights tested at 504 produce degraded results (input resolution mismatch with training); they don't reflect the model's intended deployment quality. The user's evaluation intent is always the highres model at its target resolution.

**How to apply:**
- For `*_504` eval configs, point `pretrained_model_name_or_path` to the `_highres` ckpt path.
- Output directory should match: `googlestreet_*_highres_504/` (not `googlestreet_*_504/`).
- Use existing `mv_recon_*_highres_504.yaml` configs whenever the user mentions 504 testing — don't create lowres variants.
- If unsure which weights, ASK before creating new configs.
