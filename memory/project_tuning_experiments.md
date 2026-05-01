---
name: sat architecture tuning experiments
description: Exp1-15 done + Exp16-18 designed (frozen backbone); best overall Exp6=57.46; original Pi3 (no sat) baseline=57.20 — invalidates "sat utilization" claims of Exp7-15
type: project
originSessionId: 8d83b3ca-9845-4245-adf5-af42796924d2
---
Architecture tuning study for pi3_training_sat.py, comparing modifications to the baseline satellite point prediction pipeline.

**Why:** Improving 3D reconstruction quality (δ@0.5m/1m/2m metrics) for the Google Street + satellite dataset.

**How to apply:** Check `outputs/tuning_log/results.md` for the full results table and `outputs/saved_runs/` for per-experiment model code snapshots. The detailed summary is at `outputs/tuning_log/experiment_summary.md`. Next NEW experiment number = Exp16.

## All results (sorted by Mean):
1. Exp6 (MS + DepthRefine): **57.46** ← BEST overall
2. **Pi3_ori (no sat, no DepthRefine)**: **57.20** ← 真实 baseline,使 Exp7-15 的 sat 收益可疑
3. Exp14 (MS + SatPatchInject + 统一透视exp(z)): **57.19**
4. Exp13 (MS + SatPatchInject + 正交+softplus): **56.97**
4. Exp15 (MS + SatPatchInject + 统一透视softplus): **56.81**
5. Exp12 (MS + SatRegisterInject): **56.56**
6. Exp5 (DepthRefine only): **56.47**
7. Exp7 (MS + SatInject + DepthRefine): **56.20**
8. Exp8 (MS + SatInject all tokens): **55.71**
9. Exp9 (MS + SpatialSatCrossAttn): **55.61**
10. Exp4_v2 (SatInject 100% data): **55.02**
11. Exp10 (MS + DecoderSatInject 3x): **54.97**
12. Exp3 (MultiScaleFusion): **54.74**
13. Exp11 (MS + DualBranch SatInject): **54.52**
14. Exp4 (SatInject 60%): **54.04**
15. Exp2 (Attention-Pooled MPP): **53.00**
16. Exp1 (Spatial Conv): **51.93**
17. Baseline: **45.21**

## Key findings across all rounds:
- **Token selection matters**: patch-only (Exp13=56.97) > register-only (Exp12=56.56) > all tokens mixed (Exp8=55.71). Mixing register+patch dilutes the signal.
- **Orthographic assumption unnecessary**: 去掉正交缩放先验、统一用exp(z)透视处理 (Exp14=57.19) 优于保留正交 (Exp13=56.97)
- **exp(z) > softplus(z)**: 统一透视下 exp(z) (Exp14=57.19) > softplus(z) (Exp15=56.81)
- **Simple injection > complex injection**: 单次注入 > 多级注入 > 双分支注入
- **SatInject conflicts with DepthRefine**: Exp7 (56.20) < Exp6 (57.46)
- **100 epochs overfits severely**: Exp13_v2 (100ep) = 49.69 vs Exp13 (30ep) = 56.97

## DONE: Original Pi3 (no sat) baseline (evaluated 2026-04-20)
- **Config**: `googlestreet_ori`, `pi3.models.pi3_training.Pi3`, `use_sat=false`, `freeze_encoder=true`, `load_pi3=true`, 30 epochs
- **Output**: `outputs/googlestreet_ori/ckpts/best_model/` (epoch 15, val_loss=0.0461)
- **Eval result**: δ@0.5m=34.46, δ@1m=57.54, δ@2m=79.60, **Mean=57.20**
- **Implication**: 旧 Baseline=45.21 不公平;真实公平基准是 57.20。Exp7-15 sat 注入设计相对真实基准几乎无提升 — Exp14 (57.19) 与 Pi3_ori (57.20) 在噪声范围内,Exp6 (+0.26) 增益主要来自 DepthRefine。
- **Eval log/csv**: `Pi3_eval/Pi3/outputs/googlestreet_ori/eval_stdout.txt`, `googlestreet_sat.csv`

## Pending: Exp16-18 (Frozen backbone experiments)
- **New model file**: `pi3/models/pi3_frozen_sat.py` — freezes all original Pi3 weights, only trains sat adapters
- **New config**: `configs/googlestreet_sat_frozen.yaml` + `configs/model/pi3_googlestreet_sat_frozen.yaml`
- **train script**: currently set to `config_name="googlestreet_sat_frozen"`
- **Key idea**: prevent catastrophic forgetting — ground/drone ability preserved, sat capability added
- **register_token split**: `register_token_grd` (frozen) + `register_token_sat` (trainable)
- **Exp16**: Frozen Pi3 + SatPatchInjection only (`use_sat_inject=True`)
- **Exp17**: Frozen Pi3 + SpatialSatCrossAttention (`use_sat_cross_attn=True`)
- **Exp18**: Frozen Pi3 + Decoder Adapters at layers 12,24,34 + SatInject (`use_decoder_adapters=True, use_sat_inject=True`)
- **Code snapshots**: saved in `outputs/saved_runs/exp16_frozen_satinject/`, `exp17_frozen_crossattn/`, `exp18_frozen_adapters/`
- **To run**: wait for original Pi3 training to finish, then start Exp16-18 sequentially

## Eval script modification:
- `Pi3_eval/Pi3/mv_recon/eval_sat.py` was modified to handle `sat_mpp=None` (for Exp14/15 which removed orthographic assumption)
- When `pred_sat_mpp is None`: uses point cloud-derived `sat_range` for projection test (like `eval.py`)
- Added import: `from pi3.utils.geometry import homogenize_points, se3_inverse`

## Infrastructure notes:
- Training: 7 GPUs (0-6), eval: GPU 7
- Must clear old checkpoints before switching architecture (auto_resume will try to load mismatched weights)
- Model code must be synced to eval repo: `cp pi3/models/pi3_training_sat.py /home/wangqw/video_program/Pi3_eval/Pi3/pi3/models/pi3_training_sat.py`
- For frozen_sat model: `cp pi3/models/pi3_frozen_sat.py /home/wangqw/video_program/Pi3_eval/Pi3/pi3/models/pi3_frozen_sat.py`
- Eval config checkpoint path: `Pi3_eval/Pi3/configs/evaluation/mv_recon_sat.yaml` → `pi3.pretrained_model_name_or_path`
- open3d upgraded to 0.19.0 on 2026-04-16
