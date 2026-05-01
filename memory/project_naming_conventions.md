---
name: Project naming — CrossGeo dataset, Cross3R model
description: Paper-facing names for the dataset and the best model. CrossGeo = the GoogleStreet sat+grd+uav dataset; Cross3R = the C1_no_dualcam architecture (C1_highres minus dual_camera_head — the ablation winner).
type: project
originSessionId: 2593d9fb-88bf-4bdb-9677-2c7041f59fad
---
Two project-level renames, used in writeups (paper, slides, READMEs):

**CrossGeo** = the cross-view dataset built from GoogleStreet sat + grd + uav
pairs (internal name "googlestreet_sat" / "googlestreet" everywhere in code,
configs, LMDB paths). Refer to it as **CrossGeo** in any external-facing
prose. Internal code paths/variable names stay unchanged — only the
user-facing name is CrossGeo.

**Cross3R** = the **C1_no_dualcam** model (the new best after the dual_cam
ablation showed dual_cam was harmful).
- Architecture: `pi3.models.pi3_training_sat.Pi3` with all 5 sat modules ON
  (`ablate_msfusion=False`, `ablate_satinject=False`, `ablate_satposembed=False`,
  `ablate_double_register=False`, `ablate_ortho=False`),
  `depth_activation=softplus`, **`use_dual_camera_head=False`** (single shared
  camera head, matching π³).
- Best ckpt: `outputs/saved_runs/C1_no_dualcam_highres/ckpts/best_model/pytorch_model.bin`
- Config: `configs/C1_no_dualcam_highres.yaml`
- Eval @504 on CrossGeo full test set (3 736 sequences, eval_sat_test.py):
  δ-Mean=65.44, AUC@30=76.84, PCK@2m UAV/Grd=67.75/42.60,
  MeterErr UAV/Grd=2.38/3.68 m, YawErr UAV/Grd=1.92°/3.66°.

**The previous Cross3R definition (`C1_highres`, with dual_camera_head=True)
is NOT the published model.** It is referred to as "Cross3R + dual_cam_head"
in the ablation table — the row that motivates removing dual_cam:
δ-Mean=62.34, AUC@30=77.81, PCK@2m UAV/Grd=63.74/43.65 (worse on 5/6 metrics).

**Cross3R-RotAug** (or Cross3R$_{\text{rot}}$) = the C1_satrot model — the OLD
Cross3R (with dual_cam) trained with the sat-rotation augmentation pipeline
(align sat to ground heading + U(-π,π) random spin). Best ckpt:
`outputs/saved_runs/C1_satrot_highres/ckpts/best_model/pytorch_model.bin`.
Note: this was trained before the dual_cam finding, so it includes
dual_camera_head; if reproducing, retrain with `use_dual_camera_head=False`.

**Ablation variants where only 1 or a few sat modules are added to vanilla
Pi3 are NOT Cross3R variants** — they are written as "$\pi^3$ + msfusion",
"$\pi^3$ + ortho", "$\pi^3$ + ortho + sat_pos_embed + double_register",
etc. The internal codenames (Pi3_ori_msfusion / Pi3_ori_ortho /
Pi3_ori_ortho_sp_dr — corresponding to ablations H / I / J) stay in code,
but writeups use the "$\pi^3$ + ..." form.

Conversely, **leave-one-out variants** trained for the ablation study — the
"C1_no_X" series (A/C/D/E) — were built on the OLD Cross3R definition
(with dual_cam). They are referred to as "Cross3R w/o X" in writeups
(e.g. "Cross3R w/o ortho" for E, "Cross3R w/o msfusion" for A). The
caveat that they retain the unhelpful dual_cam head is acceptable because
dual_cam is roughly orthogonal to the other modules and the qualitative
conclusions transfer cleanly to the dual_cam-free Cross3R.

**How to apply:**
- In any writeup, slide, or external doc, use **Cross3R** for the model and
  **CrossGeo** for the dataset. The published Cross3R has shared camera
  head (no dual_cam).
- In code/configs/internal logs/this memory's other files, the internal
  names (`C1_no_dualcam_highres`, `googlestreet_sat`, `C1_satrot`, `C1_no_*`,
  `Pi3_ori_*`) remain — don't rename anything on disk.
- Comparison naming pattern in writeups: "Cross3R vs Pi3$^*$ vs VGGT vs
  AerialMegaDepth" (where Pi3$^*$ = Pi3 fine-tuned on CrossGeo).
- Ablation table heading pattern:
  - "Cross3R (full)" = C1_no_dualcam_highres
  - "Cross3R + dual_cam_head" = C1_highres (motivates dual_cam removal)
  - "Cross3R w/o ortho" = C1_no_ortho_highres
  - "Cross3R w/o msfusion" = C1_no_msfusion_highres
  - ... etc
  - "$\pi^3$ + ortho" = Pi3_ori_ortho_highres
  - "$\pi^3$ + msfusion" = Pi3_ori_msfusion_highres
  - ...
  - "$\pi^3$\*" = Pi3_ori_highres

**Note on the CrossGeo full-test results**: all 10 reevals are documented in
`/home/wangqw/NeurIPS26Paper/cross3r_vs_pi3_ori.md` and use the unified
3 736-sequence test set with eval_sat_test.py / eval_test.py.
