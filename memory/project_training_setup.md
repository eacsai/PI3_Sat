---
name: training infrastructure setup
description: Key training infra details — GPU layout, eval script, config entry point, output paths, frozen model setup
type: project
originSessionId: 8d83b3ca-9845-4245-adf5-af42796924d2
---
Training infrastructure for Pi3 satellite experiments:

- **GPUs**: 8x available. Training on GPU 0-6 (7 processes), eval on GPU 7.
- **Training launch**: `accelerate launch --config_file configs/accelerate/ddp.yaml --num_processes 7 --num_machines 1 scripts/train_pi3.py train=train_pi3_lowres name=pi3_lowres`
- **Config entry**: `scripts/train_pi3.py` — config_name is set via `@hydra.main` decorator. Currently set to `googlestreet_sat_frozen` for Exp16-18.
  - `googlestreet_sat` → uses `pi3/models/pi3_training_sat.py` (full training, Exp1-15)
  - `googlestreet_sat_frozen` → uses `pi3/models/pi3_frozen_sat.py` (frozen backbone, Exp16-18)
  - `googlestreet_ori` → uses `pi3/models/pi3_training.py` (original Pi3, no sat)
- **Eval script**: `Pi3_eval/Pi3/mv_recon/eval_sat.py` — run on GPU 7. Metrics: δ@0.5m, δ@1m, δ@2m, Mean.
  - Modified to handle `sat_mpp=None` (for unified perspective models like Exp14/15)
- **Output paths**: 
  - Training logs (sat): `outputs/pi3_lowres/ckpts/log.txt`
  - Training logs (ori): `outputs/googlestreet_ori/ckpts/log.txt`
  - Best model: `outputs/pi3_lowres/ckpts/best_model/` or `outputs/googlestreet_ori/ckpts/best_model/`
  - Experiment snapshots: `outputs/saved_runs/<exp_name>/`
  - Results table: `outputs/tuning_log/results.md`
  - Full experiment summary: `outputs/tuning_log/experiment_summary.md`
- **Batch eval script**: `outputs/tuning_log/run_exp_batch.sh` — runs Exp9-11 sequentially
- **Important**: Must clear old checkpoints (`rm -rf outputs/pi3_lowres/ckpts/checkpoint_* best_model log.txt`) before switching architecture to avoid auto_resume loading mismatched weights

**Why:** Needed to quickly resume and monitor training without re-discovering paths each session.
**How to apply:** Use these paths to check training status, launch eval, or save experiment snapshots.
