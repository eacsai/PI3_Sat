---
name: C1_satrot two-stage training (sat rotation augmentation)
description: New C1 variant trained with sat-image rotation augmentation (align to ground heading + random U(-pi, pi)); replicates C1 → C1_highres pipeline; aimed at heading-robustness for KITTI/cross-view tasks
type: project
originSessionId: 2593d9fb-88bf-4bdb-9677-2c7041f59fad
---
Goal: improve C1's robustness to sat-image rotation for KITTI cross-view
where sat north-up assumption breaks. Two-stage replicates C1 → C1_highres.

## Augmentation logic (`datasets/googlestreet_dataset.py::_get_views`)

Per-sample, applies one rotation angle to ALL sat views in the sample:

```
heading = R_first_ground_c2w[:, 2]              # ground +Z in world (X-Z plane)
theta_align = atan2(heading[0], heading[2])     # align sat right axis to ground heading
phi  = U(-pi, pi)                               # extra random spin
sat_theta = theta_align + phi
# image: cv2.warpAffine(angle_deg = +degrees(sat_theta), bilinear/nearest, fill 0)
# extrinsic: R_sat_c2w_new = R_sat_c2w_orig @ R_z_cam(sat_theta)  (translation invariant)
```

Validated with debug check: with phi forced to 0, sat right axis aligns with
some ground view's heading at cos≈+1.0 / sin≈0 for all samples that have a
ground view; ones without ground keep sat unchanged (sat-only fallback).

## Files added

- `configs/data/googlestreet_sat_rotaug.yaml` — same as `googlestreet_sat.yaml`
  but `sat_rotation_aug: true` for train, `false` for test.
- `configs/C1_satrot.yaml` — lowres composition (uses model
  `pi3_googlestreet_sat_C1.yaml`, train `train_pi3_lowres_megadepth`, data
  `googlestreet_sat_rotaug.yaml`). Output: `saved_runs/C1_satrot/`.
- `configs/C1_satrot_highres.yaml` — highres composition (uses model
  `pi3_googlestreet_sat_C1_highres.yaml`, train `train_pi3_highres`, data
  `googlestreet_sat_rotaug.yaml`). `model.ckpt` overridden in `_self_` to
  point at the satrot lowres best_model. Output: `saved_runs/C1_satrot_highres/`.

## Training results

**Lowres (5 GPUs: 0,3,4,5,6 — eval was holding 1,2)**, 30 epoch × 170 iter, ~3h:

| Epoch | val (best so far) |
|-------|-------------------|
|     0 | 0.1128            |
|     1 | 0.0907            |
|     2 | 0.0768            |
|     4 | 0.0701            |
|     8 | 0.0697            |
|    15 | 0.0614            |
|  **27** | **0.0598** ← best |

**Highres (7 GPUs)**, 30 epoch × 170 iter, ~5h21m, ckpt-chained from lowres best:

| Epoch | val (best so far) |
|-------|-------------------|
|     0 | 0.1199            |
|     1 | 0.0977            |
|     2 | 0.0945            |
|     6 | 0.0923            |
|    13 | 0.0867            |
|    19 | 0.0851            |
|    21 | 0.0817            |
|    27 | 0.0811            |
|  **28** | **0.0801** ← best |

For comparison (no rotation aug): C1_highres N/A val number,
C2_highres=0.0641, Pi3_ori_highres=0.0692. **Note: val loss alone does NOT
predict test performance** — C2 had lower val but tested worse than C1 on
@504 metrics. The satrot benefit is expected to show on rotated inputs
(KITTI cross-view), not on the in-distribution googlestreet test set.

## Output paths

- Lowres ckpt: `outputs/saved_runs/C1_satrot/ckpts/best_model/pytorch_model.bin`
- Highres ckpt: `outputs/saved_runs/C1_satrot_highres/ckpts/best_model/pytorch_model.bin`
- Lowres log: `outputs/saved_runs/C1_satrot_train.log`
- Highres log: `outputs/saved_runs/C1_satrot_highres_train.log`
- WandB: project `C1_satrot` (run `fluent-leaf-1`) and `C1_satrot_highres` (run `genial-leaf-1`)

## How to apply

- For new "rotation-robust" sat experiments, copy `googlestreet_sat_rotaug.yaml`
  and set `sat_rotation_aug: true` on the train dataset only — keep test off
  so eval stays comparable to baselines.
- The augmentation is a no-op when no ground view appears in the selected
  views (sat-only or uav-only samples). All sats in a sample share the same
  rotation angle.
- Image rotation lives at the END of the per-view processing (after
  `_crop_resize_if_necessary`), so it always operates on the final-resolution
  sat at its principal-point center. Intrinsics are unchanged by the rotation.
- Eval rotation behavior is intentionally OFF — to test rotation robustness,
  run KITTI eval (`mv_recon/eval_kitti_sat.py`) which already has a
  `+kitti_rotation_range=180` knob.
