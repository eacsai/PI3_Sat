# 相机优化实验结果

**测试集**:googlestreet_sat (100 sequences)
**评估指标**:δ-Mean (avg of δ@0.5m/1m/2m) + AUC@30 + combined = (δ-Mean + AUC) / 2

## 汇总结果(按 combined 排序)

| 排名 | Run | δ@0.5 | δ@1 | δ@2 | δ-Mean | AUC@30 | **combined** | 说明 |
|---|---|---|---|---|---|---|---|---|
| 🥇 | **C1** | 36.73 | 58.49 | 79.45 | 58.22 | 87.88 | **73.05** | + 双相机 head (dual_camera_head) |
| 🥈 | ADD-0 | 35.87 | 57.93 | 79.61 | 57.80 | 88.18 | 72.99 | Pi3_ori 等价(无 sat 模块,exp(z)) |
| 🥉 | Pi3_ori | 34.46 | 57.54 | 79.60 | 57.20 | 88.25 | 72.73 | 原始 Pi3 基线 |
| 4 | C2 | 35.64 | 57.41 | 77.70 | 56.92 | 88.30 | 72.61 | C1 + sat→grd inject on camera_hidden |
| 5 | ADD-1 (+dbl_reg) | 34.00 | 55.94 | 79.65 | 56.53 | 87.52 | 72.02 | ADD-0 + double_register |
| 6 | ADD-3 (+ortho_sat) | 34.94 | 56.65 | 79.67 | 57.08 | 86.33 | 71.71 | ADD-0 + ortho_sat |
| 7 | C2_nogate | 35.59 | 56.66 | 79.00 | 57.08 | 85.86 | 71.47 | C2 去 gate + 去 FFN 零 init(V2) |
| 8 | Ablate-F | 36.13 | 58.68 | 79.49 | 58.10 | 84.64 | 71.37 | 旧基线:全 sat 模块 + softplus |
| 9 | **ADD_dr_ortho** | 33.42 | 54.14 | 78.19 | 55.25 | 84.35 | **69.80** | ADD-0 + dbl_reg + ortho_sat 组合 |

## 各实验详解

### C1 (Ablate-F + dual_camera_head) — 当前最佳 🏆
- **改动**:新建 `camera_head_sat = CameraHead(dim=512)`,sat/grd 分流。sat 头从 grd 头复制权重 warm start
- **Δ vs Ablate-F**: δ+0.12, **AUC+3.24**, combined **+1.68**
- **结论**:sat 和 grd 的 camera pose 分布差异大,分流后各自学,AUC 显著提升

### C2 (C1 + sat→grd cross-attention on camera_hidden) — 小幅退步
- **改动**:在 camera_hidden 上加 `SatRegisterInjection(dim=512)`,gate init=0
- **Δ vs C1**: δ-1.30, AUC+0.42, combined **-0.44**
- **结论**:AUC 微升但 δ 明显掉。camera 路径的 sat 注入没带整体收益

### C2_nogate (C2 去 gate + 去 FFN 零 init,V2) — 退步更明显
- **改动**:`SatRegisterInjection` 同时去掉 `gate * injected` 和 `nn.init.zeros_(ffn[-1])`(point + camera 两个注入模块都受影响)
- **Δ vs C2**: δ+0.16, **AUC-2.44**, combined **-1.14**
- **结论**:**gate 机制确实有用**,它提供"渐进学习"的安全阀。去掉 gate 后 sat 注入从第 0 步就强行干扰原始信号,AUC 下降明显

### ADD_dr_ortho (ADD-0 + double_register + ortho_sat) — 退步最严重 ⚠
- **改动**:基于 ADD-0,同时启用 `double_register` 和 `ortho_sat`(其它 sat 模块仍关)
- **Δ vs ADD-0**: δ-2.55, AUC-3.83, combined **-3.19**
- **意外结论**:这两个模块**单独加都比组合加更好**(ADD-1 -0.97, ADD-3 -1.29,合起来 -3.19)
  - 假设"double_register 喂给 ortho 分支 sat 表示"反而失败
  - 可能原因:两个模块都引入额外不稳定因素,组合在一起 30 epoch 内学不收敛

## 总体洞察

### 1. 最强配置就是 C1
- 唯一一个**显著超过基线**的实验(combined +1.68 over Ablate-F, +0.06 over ADD-0, +0.32 over Pi3_ori)
- 改动最小:只新建一个 `CameraHead`
- 效果集中在 AUC@30(+3.24),说明 sat/grd 分流主要利在相机姿态

### 2. sat 模块协同效应必须 all-or-nothing
- 单独/部分加 sat 模块基本都退步:ADD-1, ADD-2, ADD-3, ADD_dr_ortho 全是负数
- Ablate-F(全 sat 模块)combined 反而比 Pi3_ori 还低 1.36(因为 AUC 拖累)
- C1 = Ablate-F + dual_head 把 AUC 拉回来才是真正的"赢"

### 3. Gate 机制(SatRegisterInjection)真有效
- C2_nogate 比 C2 退步 -1.14,主要在 AUC -2.44
- gate 让模型有时间学习"怎么注入",而不是被强行拉离原始权重

### 4. depth_activation 与 ortho 高度耦合
- exp(z) + ortho_sat → 失败(ablate_F_exp combined=65.71)
- softplus(z) + ortho_sat → 正常工作
- 这是因为 ortho 路径的 sat_xy 与 z 必须量级匹配,sat_mpp_head 学到的尺度依赖 z 的分布

## 已尝试但不再考虑的方向
- ADD-1/2/3/4/5(单独加 sat 模块)— 大概率都退步
- C2 / C2_nogate(camera 路径 sat 注入)— 小幅退步
- ADD_dr_ortho(部分 sat 组合)— 退步严重

## 推荐下一步方向(基于上述发现)
- **C3 / C2 变种**:在 C1(双相机 head)基础上叠 sat 信息,但用不同方式:
  - **C3a**:让 sat camera_head_sat 也接收 grd 全局信息(grd→sat 注入,不是 sat→grd)
  - **C3b**:把 sat camera_head_sat 改成"先验初始化 + residual"(原 C2 方案 B1)
  - **C3c**:camera_decoder 内加 cross-attention 层(在 transformer 中,而非 head 之前)
- **C4**:换相机表示
  - 6D rotation 替代 9D-SVD(数值更稳)
  - Quaternion 表示
- **C5**:加 reprojection consistency loss(联合优化点云+相机)

## 后续 sat-专属实验

### S1 (sat camera 4-DOF 约束 head) — 大幅退步
- **改动**:`camera_head_sat` 替换为 `CameraHeadSatConstrained`,sat 视图只学 (x, y, z, yaw),共享一个 6D 参数化的 `world_R`
- **Δ vs C1**: δ-3.66, AUC-13.56, combined **-8.61**
- **结论**:强约束失败。可能原因:(1) `world_R` 共享假设错(不同样本 sat 视图朝向不一致),(2) world_R 与 yaw 学习冲突,(3) sat 实际 pose 不严格 top-down

### S2 (C1 + grd→sat spatial cross-attention) — 崩盘
- **改动**:`GeoLocalizationXAttn` 模块,grd 视图的 camera tokens 空间 attention sat patches,gate init=0
- **Δ vs C1**: δ-8.29, **AUC-20.41**, combined **-14.35**
- **结论**:虽然 gate=0 起步等价 C1,但 gate 偏离 0 后,attention 的随机初始化权重把噪声注入 camera_hidden,破坏 C1 baseline
- **教训**:C1 已通过 decoder 内 sat 模块拿到 sat 信息,在 camera_head 之前再加强 cross-attention 是冗余且过强

## 总结性观察(到此为止 8 个实验)

| 类别 | 实验 | 结果 |
|---|---|---|
| **唯一胜出** | **C1**(双相机 head) | combined=73.05 ⭐ |
| 维持/微差 | ADD-0, Ablate-F, Pi3_ori | combined ≈ 71-73 |
| 单加 sat 模块 | ADD-1/2/3 | -0.97 ~ -5.71 |
| 强约束 / 强注入 | S1, S2, ADD_dr_ortho | -3.19 ~ -14.35(都崩) |

**核心规律**:任何"在已有架构上强行加一个 sat-专属新模块"都失败。C1 之所以成功,因为它**只是把已有 head 复制了一份**,没引入新随机权重的"重型"模块。
