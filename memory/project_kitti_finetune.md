---
name: KITTI cross-view fine-tune of C1_highres
description: New training pipeline + loss for KITTI sat+ground pair fine-tuning (no point-cloud GT); first results show camera-only fine-tune plateaus and underperforms baseline
type: project
originSessionId: 2593d9fb-88bf-4bdb-9677-2c7041f59fad
---
KITTI cross-view (sat+ground pair) fine-tune of C1_highres. Built this
session, plateaued early; full results are below.

## Files added (not in C1 codepath)

**Training side** (`/home/wangqw/video_program/Pi3`):
- `datasets/kitti_pair_dataset.py` — `KittiPairDataset(BaseDataset)`. Reads
  KITTI list files (train/test1/test2), constructs sat+ground pair with
  googlestreet pose convention (sat at y=-150 via `sat_gap=150`,
  R_sat=[[0,1,0],[0,0,1],[1,0,0]] matches real LMDB sample). No depth GT →
  dummy depthmap=1 (ground), depthmap=150 (sat) so `valid_mask.sum() > 0`
  asserts pass. Extra fields: `kitti_sat_mpp`, `kitti_ground_uv_in_sat`.
- `pi3/models/loss_kitti.py` — `Pi3KittiLoss`. Wraps Pi3's `CameraLoss`
  (×0.1 mixing factor like Pi3Loss); per-batch scale = mean(|t_gt|) /
  mean(|t_pred|), detached. Optional sat-projection consistency loss
  (default `proj_weight=0` — see note below).
- `trainers/pi3_kitti_trainer.py` — `Pi3KittiTrainer(Pi3Trainer)`. With
  `cfg.train.freeze_point_branch=True`, freezes encoder/decoder/
  register_token/sat_pos_embedder/ms_fusion/point_decoder/point_head/
  sat_register_inject/sat_mpp_head — only camera_decoder + camera_head +
  camera_head_sat stay trainable. **Must access `self.model.module.X`**
  after accelerator.prepare wraps the model.
- Configs: `configs/data/kitti_pair.yaml`, `configs/general/kitti_pair*.yaml`,
  `configs/train/train_pi3_kitti.yaml`,
  `configs/model/pi3_kitti_C1_finetune{,_camonly}.yaml`,
  `configs/kitti_C1_finetune{,_camonly}.yaml`.

**Eval side** (`/home/wangqw/video_program/Pi3_eval/Pi3`):
- `mv_recon/eval_kitti_sat.py` — added `kitti_split`, `kitti_rotation_range`,
  `kitti_shift_range_lat/lon`, `pred_clip_meters` config knobs.
- `configs/evaluation/mv_recon_kitti_C1_*.yaml` — eval configs for
  baseline / KITTI-finetune / camera-only variants.

## Key design constraint: sat_mpp_head bound

`pi3/models/pi3_training_sat.py:574` clamps sat_mpp output to
`[MIN_MPP=0.005, MAX_MPP=0.05]` m/px via sigmoid. KITTI's actual sat
mpp ≈ 0.196 m/px — **4× above the model's MAX_MPP**. So sat_mpp_head
saturates at 0.05 even if trained, and the projection loss is structurally
broken on KITTI. **Disable projection loss** (`proj_weight=0`) for KITTI
fine-tune; pose loss with `_scale_align` already provides absolute-meter
supervision.

## Results: KITTI test1 first 10 samples (eval_kitti_sat.py)

| Metric | C1_highres baseline (no fine-tune) | camera-only fine-tune (best ep 4) |
|---|---|---|
| MeterError-Ground-Mean | **14.07 m** | 16.67 m |
| MeterError-Ground-Med | **11.68 m** | 18.98 m |
| YawError-Ground-Mean | 37.96° | **32.77°** |
| YawError-Ground-Med | 27.08° | 25.67° |
| Lateral-R@1m | **40 %** | 0 % |
| Longitudinal-R@1m | 0 % | **10 %** |
| Lateral-R@3m | **70 %** | 10 % |
| Longitudinal-R@3m | 0 % | **20 %** |

**Why fine-tune lost on lateral:** baseline already had a lateral prior
(40% R@1m) from googlestreet sat-cross-view training; camera-only fine-tune
plateaued at val 6.06 (epoch 4 of 30 done, no improvement after that) and
the per-batch `_scale_align` mechanism trades lateral skill for marginal
longitudinal/yaw gains.

## Open issues / not retried this session

- `_scale_align` in `Pi3KittiLoss` is detached → no absolute-scale
  gradient. Model translations grew tiny (scale → 13000+) over training.
- Camera-only freezing prevents the shared decoder from adapting to
  KITTI's coord convention; pose-loss alone may be insufficient.
- Untried: full backbone unfreeze (was the very first KITTI fine-tune;
  had `clip_loss=10` bug → loss zeroed → no learning; later runs with
  `clip_loss=2000` showed loss decreasing but plateaued ~val 34).

## Training paths

- Output: `outputs/saved_runs/kitti_C1_finetune{,_camonly}/`
- Best ckpt: `outputs/saved_runs/kitti_C1_finetune_camonly/ckpts/best_model/pytorch_model.bin`
- Train log: `outputs/saved_runs/kitti_C1_finetune_camonly_train.log` (stdout)
- Hydra log: `outputs/saved_runs/kitti_C1_finetune_camonly/log.log`
- WandB project: `pi3_kitti_C1_finetune` (innov1ise-shanghaitech-university)

## How to apply
- For new KITTI experiments, start from `kitti_C1_finetune_camonly.yaml`
  and tune the freeze list / loss weights via Pi3KittiTrainer flags.
- Don't enable `proj_weight > 0` until sat_mpp_head sigmoid bound is widened.
- Use `eval_kitti_sat.py` with `kitti_split=test1` to test (3773 pairs);
  for quick checks pass `+limit=10`.
