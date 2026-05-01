---
name: train.clip_loss=10 silently zeros large initial losses
description: BaseTrainer multiplies loss by 0 if it exceeds clip_loss; default 10 is too low for fine-tunes whose initial loss is in the hundreds — train shows loss=0 but val shows real numbers
type: feedback
originSessionId: 2593d9fb-88bf-4bdb-9677-2c7041f59fad
---
`trainers/base_trainer_accelerate.py:351-352`:

```python
if loss > self.cfg.train.clip_loss:
    loss = loss * 0.0
```

The whole loss tensor is zeroed (not clipped to `clip_loss`!), so the
backward pass produces zero gradients and nothing learns.

**Why this is sneaky:** validation runs the loss without the clip, so the
val log shows real numbers (e.g. KITTI val=810 → 71 dropping nicely) while
the train log shows `loss: 0.0000 (0.0000)` and `trans_loss: 0.0000` etc.
across all components. Easy to mistake for "loss already converged".

**When this hits:**
- Fine-tunes from a strong-prior model into a new domain (e.g. C1_highres
  → KITTI), where initial loss can legitimately be hundreds before scale
  alignment kicks in.
- Loss formulations without point-cloud normalisation, where translations
  in absolute meters dominate (Pi3KittiLoss without `*0.1` mixing).

**How to apply:**
- For new fine-tunes / domain-shifts, set `train.clip_loss` ≥ ~10× the
  expected initial loss (e.g. `clip_loss: 2000` for KITTI sat+ground
  pose-loss in raw meters with `trans_alpha=100`).
- If a training run shows train `loss: 0.0000` but val is non-zero, the
  first thing to check is `clip_loss` vs the val loss range.
- Only fall back to small `clip_loss` when initial loss is already in the
  ~1 range (point-cloud-normalised Pi3Loss on googlestreet).
