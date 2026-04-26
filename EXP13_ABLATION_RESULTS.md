# Exp13 模块消融实验结果

**Baseline**: Exp13 (MultiScaleFusion + SatPatchInjection + sat_pos_embed + double register + ortho sat branch + loss_sat.py)
= full finetune 30 epoch on 100% data, **Mean = 56.97**

**参考**:Pi3_ori (无 sat 模块, loss.py) Mean = **57.20**

每个消融实验**只关闭一个**组件,其它保持 Exp13 设置。

## 主结果表

| 编号 | 关闭组件 | δ@0.5m | δ@1m | δ@2m | **Mean** | vs Exp13 |
|---|---|---|---|---|---|---|
| **Exp13** | (baseline) | 34.66 | 56.96 | 79.30 | **56.97** | — |
| Pi3_ori | (无 sat 模块,reference) | 34.46 | 57.54 | 79.60 | **57.20** | +0.23 |
| Ablate-A | MultiScaleFusion (gated 4-layer fusion) | 28.42 | 50.58 | 75.73 | **51.58** | -5.39 |
| Ablate-B | SatPatchInjection (sat→grd 全局注入) | 34.60 | 58.61 | 80.00 | **57.74** | +0.77 |
| Ablate-C | Fourier sat 位置编码 | 35.46 | 58.31 | 79.33 | **57.70** | +0.73 |
| Ablate-D | 双 register_token (sat / grd 各一组) | 29.06 | 49.93 | 75.01 | **51.33** | -5.64 |
| Ablate-E | 正交 sat 分支 + sat_mpp_head | 30.30 | 53.40 | 77.72 | **53.80** | -3.17 |
| Ablate-F | loss_sat.py → loss.py (与 Pi3_ori 同) | 36.13 | 58.68 | 79.49 | **58.10** | +1.13 |
| Ablate-G | normal_loss 在 train 和 eval 都关闭 | 32.95 | 55.28 | 78.13 | **55.45** | -1.52 |

## 各 ablation 详细动作

| 编号 | flag/change | 含义 | 实际动作 |
|---|---|---|---|
| Ablate-A | `ablate_msfusion` | MultiScaleFusion (gated 4-layer fusion) | 改回 last-2 layers concat (原 Pi3) |
| Ablate-B | `ablate_satinject` | SatPatchInjection (sat→grd 全局注入) | 完全跳过 sat_register_inject 调用 |
| Ablate-C | `ablate_satposembed` | Fourier sat 位置编码 | decoder 入口不加 sat_pos_embed |
| Ablate-D | `ablate_double_register` | 双 register_token (sat / grd 各一组) | 改回单组 register_token (原 Pi3) |
| Ablate-E | `ablate_ortho` | 正交 sat 分支 + sat_mpp_head | sat 用透视 xy*z (与 grd 一致) |
| Ablate-F | `loss=loss.py` | loss_sat.py → loss.py (与 Pi3_ori 同) | 去掉 per-view weight 归一化 + sat orthogonal loss |
| Ablate-G | `normal_loss off` | normal_loss 在 train 和 eval 都关闭 | normal_loss_start_epoch=999 → 30 epoch 训练内永远不触发 |

## 完整指标(参照表)

| 编号 | Acc-mean | Comp-mean | AUC@30 | UAV-Meter | Ground-Meter | UAV-Yaw | Ground-Yaw |
|---|---|---|---|---|---|---|---|
| Ablate-A | 1.493 | 1.697 | 84.02 | 2.46 m | 4.98 m | 2.66° | 3.24° |
| Ablate-B | 1.303 | 1.525 | 87.48 | 1.71 m | 2.90 m | 1.66° | 2.07° |
| Ablate-C | 1.340 | 1.614 | 83.22 | 1.93 m | 3.40 m | 1.44° | 6.47° |
| Ablate-D | 1.585 | 1.877 | 71.08 | 1.96 m | 5.62 m | 2.48° | 30.84° |
| Ablate-E | 1.409 | 1.552 | 82.21 | 4.39 m | 4.99 m | 1.80° | 7.13° |
| Ablate-F | 1.319 | 1.503 | 84.64 | 1.92 m | 3.22 m | 1.53° | 7.72° |
| Ablate-G | 1.415 | 1.538 | 78.99 | 2.36 m | 4.40 m | 3.46° | 8.74° |

## 关键解读

- **Δ Mean 越负 = 该组件贡献越大**(去掉后效果掉得越多)
- **Δ Mean 正值 = 该组件可能是负贡献**(去掉反而更好)

**组件重要性排序(掉分越多 = 越重要)**:
1. **Ablate-D** (双 register_token (sat / grd 各一组)) → Mean = 51.33 (Δ -5.64)
2. **Ablate-A** (MultiScaleFusion (gated 4-layer fusion)) → Mean = 51.58 (Δ -5.39)
3. **Ablate-E** (正交 sat 分支 + sat_mpp_head) → Mean = 53.80 (Δ -3.17)
4. **Ablate-G** (normal_loss 在 train 和 eval 都关闭) → Mean = 55.45 (Δ -1.52)
5. **Ablate-C** (Fourier sat 位置编码) → Mean = 57.70 (Δ +0.73)
6. **Ablate-B** (SatPatchInjection (sat→grd 全局注入)) → Mean = 57.74 (Δ +0.77)
7. **Ablate-F** (loss_sat.py → loss.py (与 Pi3_ori 同)) → Mean = 58.10 (Δ +1.13)

---

**生成脚本**: `outputs/googlestreet_sat/gen_ablate_summary.py`
**详细日志**: `outputs/googlestreet_sat/orch_ablate.log`
**各实验快照**: `outputs/saved_runs/exp13_ablate_{A,B,C,D,E,F}/`