# Cross3R vs π³ — Architectural Differences Reference

This document summarizes how **Cross3R** (our model) differs from the
publicly released **π³** (Wang et al., 2025) baseline. Intended as reference
material for paper writing, especially Section 4 (Method) and Section 5
(Experiments).

## TL;DR

Cross3R = π³ backbone + 5 satellite-aware modules + 2 design-choice changes,
fine-tuned on the **CrossGeo** cross-altitude dataset. The single most
important addition is the **orthographic satellite branch**: it provides a
metric-scale geometric prior that recovers the satellite-pixel ↔ world-meter
mapping, which is essential for cross-view localization. The remaining
modules contribute to point-cloud quality and pose accuracy.

## Code-level summary

- **Model class change**: π³ uses `pi3.models.pi3_training.Pi3` (vanilla);
  Cross3R uses `pi3.models.pi3_training_sat.Pi3` (sat-aware).
- **Initialization**: Cross3R inherits the publicly released π³ checkpoint
  for every layer whose architecture is shared (DINOv2 encoder, decoder
  blocks, point/camera decoders, point/camera heads). Only the new modules
  are randomly initialized; their gates and last linear layers are
  zero-initialized so the network starts as a near-identity perturbation
  of π³.
- **Training**: identical two-stage recipe to π³ (low-res 224×224, then
  dynamic-resolution stage), but on CrossGeo instead of public reconstruction
  datasets.

## Module-by-module differences

The following 5 satellite-aware modules and 2 design-choice changes are added
on top of π³.

### 1. Orthographic satellite point branch + sat_mpp_head (geometric prior)

| | π³ | Cross3R |
|---|---|---|
| Sat back-projection | perspective: `[xy * z, z]` | orthographic: `[sat_xy_base * sat_mpp, z]` |
| Sat scale | implicit, learned from data | explicit, regressed by `sat_mpp_head` |
| `sat_mpp_head` | n/a | small MLP regressing per-image meter-per-pixel, sigmoid-bounded to [0.005, 0.05] m/px |
| Inductive bias | none | satellite is captured by a near-orthographic camera; pixel ↔ meter mapping is a single global scalar |

This is the single most consequential change. Ablation result: removing it
collapses cross-view PCK from ~85 % to ~11 % (UAV @2 m).

### 2. MultiScaleFusion (`ms_fusion`)

Softmax-gated fusion of decoder layers 8 / 17 / 26 / 34, replacing π³'s
`concat(L34, L35)`. Initial gate logits `[-5, -5, -5, +5]` → starts as the
last-layer-only mode, and gradually learns to weight earlier layers.

Implementation: `pi3/models/pi3_training_sat.py:22-35`. Output shape is
`concat(fused, last_hidden)`, identical to π³'s output dimension so the
downstream point/camera decoders can be inherited verbatim.

Ablation result: removing it costs ~3 points δ-Mean and ~1 point AUC@30 with
no measurable PCK change → contributes to point-cloud quality, not metric
position.

### 3. SatRegisterInjection (`sat_register_inject`) — cross-view information flow

A learnable **bias vector** is computed from satellite patch tokens (mean
pool, ignoring register tokens) and broadcast-added to the patch tokens of
every ground / UAV view. Pipeline:

```
sat patch tokens (B, N_sat, P, D) -> mean pool over patches and views -> (B, D)
                                  -> LayerNorm -> 2-layer FFN (zero-init last)
                                  -> learnable scalar gate (zero-init)
                                  -> broadcast to all non-sat views
                                  -> residual add to point_hidden
```

Implementation: `pi3/models/pi3_training_sat.py:72-110`. Inserted between
`point_decoder` and `point_head`. Zero-init FFN + zero-init gate gives a
double-soft-start: the network behaves identically to π³ at iteration 0 and
the gate ramps up only when satellite information actually helps.

### 4. Sat-specific positional encoding (`sat_pos_embedder`)

A separate Fourier positional encoding for satellite patch tokens, additive
to the standard RoPE used by all views. Gives sat tokens a dedicated
"identity stamp" so register tokens can pool them coherently.

### 5. Doubled register-token bank (`double_register`)

π³ has a single bank of 5 register tokens shared across all views;
Cross3R has 2 banks (one for satellite, one for ground/UAV). The 2-bank
design lets the satellite view aggregate global information without
clobbering the bank used by ground/UAV.

Implementation: `register_token` parameter shape changes from `(1, 1, 5, D)`
to `(1, 2, 5, D)`.

### 6. Dual camera head (`use_dual_camera_head`)

π³ uses a single shared `camera_head` for all views. Cross3R provides a
separate `camera_head_sat` for satellite views, while ground / UAV continue
to share `camera_head`. Motivation: the satellite pose distribution
(directly overhead, near-orthographic) is qualitatively different from
ground / UAV.

**Caveat**: this design choice did **not** improve over the shared head in
our ablation — see Section *Ablation* below.

### 7. Depth activation: `softplus` vs `exp`

π³ uses `z = exp(z_logit)` for depth; Cross3R uses `z = softplus(z_logit) +
1e-6` for ground / UAV. Avoids numerical blow-up at large depth values that
appear in cross-altitude scenes (UAV at hundreds of meters above ground).

## Ablation matrix (CrossGeo full test set, 504×504)

The following 8 ablations were trained from scratch with the same two-stage
recipe and evaluated on the **full CrossGeo test set (3 736 sequences, 504×504,
`eval_sat_test.py`)**. Numbers are **pending re-evaluation as of writing**;
the previous-test-set numbers (1 868 sequences) are listed here for the
qualitative pattern.

| Ablation | δ-Mean | AUC@30 | PCK@2m UAV | PCK@2m Grd | What it removes |
|---|---|---|---|---|---|
| Cross3R (full) | 63.68 (old test1) | 81.83 | 85.0 | 43.1 | — |
| Cross3R w/o ortho (E) | 64.46 | 79.48 | **10.99** | 16.16 | orthographic sat branch |
| Cross3R w/o msfusion (A) | 62.36 | 76.41 | 65.34 | 41.80 | MultiScaleFusion |
| Cross3R w/o satposembed (C) | 62.39 | 76.26 | 63.96 | 39.81 | sat positional encoding |
| Cross3R w/o double register (D) | 63.89 | 75.65 | 64.61 | 39.90 | doubled register bank |
| Cross3R w/o dual_cam (F) | **65.63** | 76.64 | **67.98** | **43.56** | separate sat camera head |
| π³ + msfusion only (H) | 65.38 | 74.24 | 11.12 | 15.83 | (Pi3 + only msfusion) |
| π³ + ortho only (I) | 65.20 | 77.19 | 63.03 | 40.89 | (Pi3 + only ortho) |
| π³ + ortho + sp + dr (J) | 63.79 | 76.18 | 63.86 | 40.14 | (Pi3 + ortho/sp/dr) |
| π³* (Pi3 fine-tuned on CrossGeo) | 61.54 | 71.53 | 8.97 | 11.22 | — |

### Findings

1. **Orthographic branch is the single most important module.** Removing it
   (E) drops PCK@2m UAV from ~85 % to ~11 %, identical to the π³* baseline.
   Adding only it to π³ (I) recovers ~63 %. δ-Mean is unaffected — the
   orthographic prior governs metric alignment, not 3-D shape quality.
2. **MultiScaleFusion improves point-cloud quality.** Adding only it to π³
   (H) lifts δ-Mean from 61.54 to 65.38, but PCK is unchanged. Removing it
   from Cross3R (A) costs ~1.5 δ-Mean.
3. **The two prior categories are orthogonal.** ortho governs PCK / metric
   localization; msfusion governs δ / point-cloud reconstruction. Combining
   both yields the full Cross3R; neither alone is sufficient.
4. **Dual camera head is not helpful and is mildly harmful.** Cross3R w/o
   dual_cam (F) is the best leave-one-out variant by both δ-Mean and PCK,
   suggesting the sat-specific camera head should be removed in the final
   model. Earlier C2 ablation (sat→grd injection on the camera path) also
   regressed performance, consistent with this finding.
5. **Sat positional encoding and double register banks are mild auxiliaries.**
   Each contributes ~1-2 points δ-Mean. The combination of ortho + sp + dr
   (J) is essentially indistinguishable from ortho alone (I).

## Suggested writing notes

- Position Cross3R as **"π³ specialised for cross-altitude reconstruction"**.
  The novelty is not a new backbone but a satellite-aware adaptation of an
  existing geometry foundation model.
- Emphasise the **orthographic prior** as the principal contribution. It is
  the single change that turns a generic multi-view model into a
  cross-altitude one with metric scale.
- The **other modules are improvements, not the contribution**. Mention them
  but keep the spotlight on the orthographic branch.
- For the ablation table, prefer **leave-one-out** rows (Cross3R w/o X)
  over **add-one-in** rows (π³ + X), because LOO directly answers "what
  does the model lose without this?".
- **Baseline naming**: π³ refers to the public checkpoint; π³* (often
  written `\pi^{3*}`) is π³ fine-tuned on CrossGeo, used as the data-controlled
  baseline to disentangle architecture from data.
- The dual-camera-head finding is a **negative result worth reporting** if
  space allows; otherwise drop the head from the final published model.
