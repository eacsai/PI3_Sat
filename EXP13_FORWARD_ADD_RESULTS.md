# Forward-Add 贪心模块累加消融

**思路**:从 Pi3_ori 等价基线(无 sat 模块,loss.py)出发,按之前 ablation 测出的重要性顺序逐一尝试加入 Exp13 模块。每加一个就训 30 epoch + eval。

**决策规则**:Mean 提升 > 0.10 → 保留该模块,否则丢弃,继续测下一个候选模块

**参考线**:
- Pi3_ori (零 sat 模块): Mean = **57.2**
- Exp13 (全 sat 模块 + loss_sat.py): Mean = **56.97**
- Ablate-F (全 sat 模块 + loss.py): Mean = **58.1**

## 候选模块顺序(按 ablation 重要性)

| 顺位 | 模块 | 描述 |
|---|---|---|
| 1 | `double_register` | 双 register_token (sat / grd 各一组) |
| 2 | `ms_fusion` | MultiScaleFusion (4 层 gated 融合) |
| 3 | `ortho_sat` | 正交 sat 分支 + sat_mpp_head |
| 4 | `sat_pos_embed` | Fourier sat 位置编码 |
| 5 | `sat_patch_inject` | SatPatchInjection (sat→grd 注入) |

## 实验进展

| 步骤 | 候选模块 | 试验中 active | Mean | Δ vs 上轮 best | 决策 |
|---|---|---|---|---|---|
| ADD-5_sat_patch_inject | (none) | (empty) | Wrote /home/wangqw/video_program/Pi3/configs/model/pi3_googlestreet_sat_ablate_add_5.yaml (active modules: ['ablate_satinject']) | — | — |
| ADD-2_ms_fusion | (none) | (empty) | Wrote /home/wangqw/video_program/Pi3/configs/model/pi3_googlestreet_sat_ablate_add_2.yaml (active modules: ['ablate_msfusion']) | — | — |
| ADD-4_sat_pos_embed | (none) | (empty) | [2026-04-22 22:05:36] === [ADD-4_sat_pos_embed] START (active modules: ablate_satposembed) === | — | — |
| Wrote /home/wangqw/video_program/Pi3/configs/model/pi3_googlestreet_sat_ablate_add_4.yaml (active modules | (none) | (empty) | — | — | — |
| [2026-04-22 22 | (none) | (empty) | — | — | — |
| [2026-04-22 22 | (none) | (empty) | — | — | — |
| ADD-0 | (none) | (empty) | Wrote /home/wangqw/video_program/Pi3/configs/model/pi3_googlestreet_sat_ablate_add_0.yaml (active modules: NONE (Pi3_ori-like)) | — | — |
| ADD-3_ortho_sat | (none) | (empty) | Wrote /home/wangqw/video_program/Pi3/configs/model/pi3_googlestreet_sat_ablate_add_3.yaml (active modules: ['ablate_ortho']) | — | — |
| torch.distributed.elastic.multiprocessing.errors.ChildFailedError | (none) | (empty) | — | — | — |
| Failures | (none) | (empty) | — | — | — |
| Root Cause (first observed failure) | (none) | (empty) | — | — | — |
| [0] | (none) | (empty) | — | — | — |
| time | (none) | (empty) | — | — | — |
| host | (none) | (empty) | — | — | — |
| rank | (none) | (empty) | — | — | — |
| exitcode | (none) | (empty) | — | — | — |
| error_file | (none) | (empty) | — | — | — |
| traceback | (none) | (empty) | — | — | — |

## 完整 eval 指标

| 步骤 | δ@0.5m | δ@1m | δ@2m | Mean | Acc-mean | AUC@30 | UAV-Meter | Ground-Meter |
|---|---|---|---|---|---|---|---|---|
| ADD-5_sat_patch_inject | — | — | — | — | — | — | — | — |
| ADD-2_ms_fusion | — | — | — | — | — | — | — | — |
| ADD-4_sat_pos_embed | — | — | — | — | — | — | — | — |
| Wrote /home/wangqw/video_program/Pi3/configs/model/pi3_googlestreet_sat_ablate_add_4.yaml (active modules | — | — | — | — | — | — | — | — |
| [2026-04-22 22 | — | — | — | — | — | — | — | — |
| [2026-04-22 22 | — | — | — | — | — | — | — | — |
| ADD-0 | — | — | — | — | — | — | — | — |
| ADD-3_ortho_sat | — | — | — | — | — | — | — | — |
| torch.distributed.elastic.multiprocessing.errors.ChildFailedError | — | — | — | — | — | — | — | — |
| Failures | — | — | — | — | — | — | — | — |
| Root Cause (first observed failure) | — | — | — | — | — | — | — | — |
| [0] | — | — | — | — | — | — | — | — |
| time | — | — | — | — | — | — | — | — |
| host | — | — | — | — | — | — | — | — |
| rank | — | — | — | — | — | — | — | — |
| exitcode | — | — | — | — | — | — | — | — |
| error_file | — | — | — | — | — | — | — | — |
| traceback | — | — | — | — | — | — | — | — |

## 解读

- 每步都从基线(Pi3_ori 等价 + 已保留模块)出发,**只加一个**新候选模块训 30 epoch
- 若新模块对 Mean 没明显提升(< 0.1),就**丢弃它**,直接进下一个候选
- 这与之前的 "先 Exp13 全模块,逐个去掉" 互补:
  - 之前测的是 "模块对 Exp13 整体的贡献"
  - 这次测的是 "模块对 Pi3_ori 的边际贡献"

---

**生成脚本**: `outputs/googlestreet_sat/gen_forward_add_summary.py`
**详细日志**: `outputs/googlestreet_sat/orch_forward_add.log`
**各 step 快照**: `outputs/saved_runs/forward_add/`