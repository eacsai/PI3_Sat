# Pi3 training project memory

## Audit identity and authority

- Authoritative host: wangqw@10.15.89.238 (fineserver), hostname observed as fineserver.
- Authoritative repository: /home/wangqw/video_program/Pi3.
- Repository branch at audit time: training, tracking origin/training, ahead by 4 commits.
- This file is project memory for the NeurIPS 2026 rebuttal audit. It is not a replacement
  for the source code or the experiment logs.
- Audit date: 2026-07-23 (Asia/Shanghai).
- The read-through covered the repository's source, Python, shell, YAML, Markdown, and other
  text/configuration files in the read-only audit snapshot, including active entry points,
  datasets, models, losses, trainers, configs, launch queues, and experiment evidence.
  Datasets, checkpoints, caches, generated outputs, media, and binary artifacts were not
  copied into the snapshot or treated as line-readable source. Their locations were checked
  separately on the authoritative host.

## Paper-to-code baseline

The paper is 5620_Seeing_Across_Skies_and_S.pdf, titled “Seeing Across Skies and
Streets: Feedforward 3D Reconstruction from Satellite, Drone, and Ground Images.”
The codebase is the Cross3R/Pi3 training side of the paper:

- CrossGeo is the six-view satellite/UAV/ground dataset described in the paper. The paper
  reports 46,302 samples, 277,812 images, and 85 scenes; the split is 75 train scenes
  (38,962 samples), 5 validation scenes (3,614), and 5 test scenes (the paper reports
  3,726 in one place and 3,736 in Table 6).
- The paper uses 224-pixel stage 1 training followed by dynamic high-resolution stage 2.
  Stage 1 is 224² with up to 3 images and per-GPU batch 64; stage 2 samples 100k–255k
  pixels with aspect ratio 0.5–2.0, uses up to 24 images, and uses the 8-GPU L40 setup
  described in the paper.
- The canonical released-Pi3 initialization is extended with satellite positional encoding,
  doubled register tokens, multi-scale feature fusion, satellite-register injection,
  orthographic satellite geometry, and a satellite scale/meter-per-pixel head. The paper
  describes this as roughly +2.5M parameters.
- Paper loss weights are normal 1, camera 0.05, and point 0.1, with normal loss warmed up
  after five epochs. The exact active implementation is Pi3Loss in the training model
  path; the dual-satellite branch has a separate loss_dualsat.py.

## Canonical training entry points

Primary launcher:

- scripts/train_pi3.py:9-12 creates a Hydra app with config_path="../configs" and
  default config_name="googlestreet_sat", then calls the configured trainer's eval
  and train methods.
- CLAUDE.md:11-19 documents the repository's intended launcher and the two-stage workflow.
  CLAUDE.md:25-31 describes the config layout; CLAUDE.md:35-55 records architecture
  assumptions and common gotchas.
- The stock README command is not sufficient for every experiment: the launcher decorator
  defaults to googlestreet_sat, while the experiment queues pass the experiment name
  through --config-name. Always use the queue's explicit config name or verify the
  decorator before running.

Canonical Cross3R run:

    cd /home/wangqw/video_program/Pi3
    CUDA_VISIBLE_DEVICES=0 python scripts/train_pi3.py --config-name=C1_no_dualcam
    CUDA_VISIBLE_DEVICES=0 python scripts/train_pi3.py --config-name=C1_no_dualcam_highres

The high-resolution config initializes from the completed low-resolution checkpoint:

/home/wangqw/video_program/Pi3/outputs/saved_runs/C1_no_dualcam/ckpts/best_model/pytorch_model.bin

The exact high-resolution config is configs/C1_no_dualcam_highres.yaml. It sets dynamic
resolution, image_num [2, 3] for the CrossGeo train contract, max_img_per_gpu 24,
pixel range [100000, 255000], iters 170, epochs 30, and writes the run under
outputs/saved_runs/C1_no_dualcam_highres.

The corresponding low-resolution config is configs/C1_no_dualcam.yaml; its model config
is configs/model/pi3_googlestreet_sat_C1_no_dualcam.yaml, its trainer config is the
low-resolution MegaDepth-style trainer, and its output name is saved_runs/C1_no_dualcam.

## Config and optimizer contract

The default CrossGeo data config is configs/data/googlestreet_sat.yaml:

- root: /data/zhongyao/dataset/
- train dataset class: datasets.googlestreet_dataset.GoogleStreetDataset
- train weight/nominal length: 39,000
- test weight/nominal length: 450
- frame_num: 3, satellite mode enabled, shift range ±20 m
- satellite gap 150 m, z-far 30,000, jitter augmentation, and focal augmentation.

The low-resolution trainer is configs/train/train_pi3_lowres_megadepth.yaml: resolution
224, image_num [2, 3], and maximum images per GPU 64. The high-resolution trainer is
configs/train/train_pi3_highres.yaml: dynamic resolution, pixel range 100,000–255,000,
aspect ratio 0.5–2.0, patch size 14, image_num [2, 24], and maximum images per GPU 24.
The key is spelled random_reslution in the existing high-resolution config; do not silently
rename it when reproducing an existing run.

configs/model/pi3_googlestreet_sat_C1_no_dualcam.yaml is the canonical model/loss contract:

- target is pi3.models.pi3_training_sat.Pi3;
- Pi3 initialization is loaded and the encoder is frozen;
- the five Cross3R ablation switches are set by the C1 config, with the canonical C1
  configuration using the Cross3R additions and the paper's shared camera head
  (use_dual_camera_head: false);
- Softplus is enabled for the satellite scale head;
- the loss is Pi3Loss;
- normal warmup is 5 epochs;
- mixed precision is bf16;
- AdamW uses learning rate 5e-5, encoder learning rate 5e-6, weight decay 0.05,
  OneCycle/cosine scheduling, gradient clipping 10, and loss clipping 10;
- auto_resume: true means an existing run directory can change the starting state.
  Record whether a run was fresh or resumed when making rebuttal claims.

trainers/base_trainer_accelerate.py is the checkpoint/schedule authority. The scheduler
uses epochs * iters as the total step budget, auto_resume restores state, periodic
states are saved under the run's ckpts, and the best state is saved using
accelerator.save_state under ckpts/best_model. If a loss exceeds clip_loss, the
trainer can zero that loss; gradient norm is clipped to 10. trainers/pi3_trainer.py
adds the separate encoder learning rate and handles dynamic samplers/resolutions,
variable-view stacking, and the is_sat_mask.

## Dataset implementation and actual paths

datasets/googlestreet_dataset.py is the source of the train/test sample contract.
Important details:

- The configured data root /data/zhongyao/dataset/ controls folder traversal, but the
  LMDB path is hard-coded as /home/wangqw/NeurIPS26/dataset_lmdb_v2.
- The loader requires the LMDB environment metadata keys __VERSION__ and __DIR_CACHE__.
  A missing LMDB or mismatched metadata is a data-environment failure, not a model failure.
- The source has sat_height=5726 and sat_gap=150. The satellite crop gap and image
  geometry therefore cannot be inferred from the YAML alone.
- data_pct is a deterministic prefix selection, not a fresh random subset. The train
  loader excludes 0516_pair, uses the configured target suffixes, and chooses east/south
  shifts in the ±20 m range.
- Current satellite crop meters are sampled uniformly in the implementation's 70–210 m
  range. The split logic distinguishes the _1 and _2 pair views.
- datasets/base/base_dataset.py converts depth to absolute camera coordinates. With
  override_sat_height enabled it places the satellite camera at y=-sat_gap; with it
  disabled it keeps the raw camera height. This switch is the code path behind the
  raw-height ablation.

Do not use a local path or a copied snapshot as evidence that the dataset is present on the
server. Verify the two absolute paths above on fineserver before a new run.

## Model and loss source map

- pi3/models/pi3_training_sat.py is the active generic training implementation used by the
  canonical config. The evaluation repository contains a byte-identical copy at the time of
  this audit; treat the checkpoint and this model file as a cross-repository contract.
- pi3/models/cross3r.py is the clean Cross3R implementation and is the best source map for
  the paper's architecture: DINOv2 ViT-L/14, 36-layer decoder with width 1024, two banks
  of five register tokens, Fourier satellite position features, multi-scale fusion at
  decoder layers 8/17/26/34, satellite register injection, a bounded satellite
  meter-per-pixel head, orthographic local satellite x/y from image coordinates times mpp,
  and a shared camera head.
- pi3/models/pi3_training_sat.py is the implementation used by the C1 configs, so use its
  exact tensor flow when answering rebuttal questions about the released run. Do not
  substitute the clean cross3r.py class name for the active Hydra target without checking
  the config.
- pi3/models/loss.py contains scale-aligned point loss, local point loss, and normal loss
  after the warmup, plus Huber-style camera translation/rotation loss. Pi3Loss combines
  point and camera terms. pi3/models/loss_dualsat.py is only for the special dual-satellite
  branch.

## Experiment matrix and paper correspondence

### Main model and Table 6 ablations

The paper's canonical Cross3R result is the high-resolution checkpoint from
C1_no_dualcam_highres, not the older C1_highres. Map paper labels as follows:

| Paper label | Training config/checkpoint | Reproduction note |
| --- | --- | --- |
| Cross3R/main | C1_no_dualcam_highres | canonical shared camera head; initialize stage 2 from C1_no_dualcam |
| w/o orthographic | C1_no_ortho_highres | corresponding C1_no_ortho low-res init |
| w/o multi-scale fusion | C1_no_msfusion_highres | corresponding low-res init |
| w/o satellite position | C1_no_satposembed_highres | corresponding low-res init |
| w/o satellite injection | C1_no_satinject_highres | corresponding low-res init |
| w/o doubled registers | C1_no_double_reg_highres | corresponding low-res init |
| w/o altitude / raw satellite height | Cross3R_rawh_highres | uses override_sat_height: false; verify eval config |
| w/o UAV / w/o ground | no separate training checkpoint | evaluation view selection/modality ablation; use matching eval config |
| old dual-camera comparison | C1_highres | historical dual-camera run; not the canonical Cross3R checkpoint |

The queue scripts/run_ablation_queue.sh enumerates the main ablations:
C1_no_ortho, Pi3_ori_ortho, C1_no_msfusion, Pi3_ori_msfusion,
Pi3_ori_ortho_sp_dr, C1_no_satposembed, C1_no_double_reg, and
C1_no_dualcam. It trains low resolution, trains high resolution, and then calls the
evaluation repository on GPU 7. The training logs are:

outputs/saved_runs/<EXP>_train.log
outputs/saved_runs/<EXP>_highres_train.log
outputs/saved_runs/<EXP>_eval504.log

scripts/run_satinject_queue.sh is the separate C1-no-satellite-injection queue. It uses
the same low-res/high-res/eval structure. scripts/run_rawh_queue.sh is the raw-height
queue and writes <EXP>_highres_reeval.log for the re-evaluation path.

### Data scaling

Cross3R_data20, Cross3R_data40, Cross3R_data60, Cross3R_data80, and Cross3R_data100
implement deterministic prefix scaling through data_pct values 0.2/0.4/0.6/0.8/1.0.
The configs adjust epochs and iters so the total budget is approximately 5,100 optimizer
steps; for example, the 20% config records iters: 30 and epochs: 170, while other
configs use their own product-preserving values. Do not report “same number of epochs”
for this table; report the actual config and total-step product.

### Satellite gap and dual-satellite variants

Cross3R_satgap100 and Cross3R_satgap200 are the code variants around the default
150 m satellite gap. The paper's supplement labels the 100/150/200 m comparison, while
the code names only the two non-default configs; preserve this distinction in a rebuttal.
Cross3R_dualsat and Cross3R_dualsat_highres use the untracked dual-satellite model and
loss files currently present in the working tree. They are not the canonical C1 checkpoint.

## Checkpoints and output roots

Public/base initialization:

/home/wangqw/video_program/Pi3/ckpts/Pi3/model_pi3.safetensors

Observed best checkpoints under /home/wangqw/video_program/Pi3/outputs/ (each path ends
in ckpts/best_model/pytorch_model.bin):

googlestreet_ori, saved_runs/C1, saved_runs/C1_highres,
saved_runs/C1_no_double_reg_highres, saved_runs/C1_no_dualcam_highres,
saved_runs/C1_no_msfusion_highres, saved_runs/C1_no_ortho_highres,
saved_runs/C1_no_satinject_highres, saved_runs/C1_no_satposembed_highres,
saved_runs/C1_satrot_highres, saved_runs/C2_highres,
saved_runs/Cross3R_data100, saved_runs/Cross3R_data20,
saved_runs/Cross3R_data40, saved_runs/Cross3R_data60,
saved_runs/Cross3R_data80, saved_runs/Cross3R_dualsat,
saved_runs/Cross3R_dualsat_highres, saved_runs/Cross3R_rawh,
saved_runs/Cross3R_rawh_highres, saved_runs/Cross3R_satgap100,
saved_runs/Cross3R_satgap200, saved_runs/Pi3_ori_highres,
saved_runs/Pi3_ori_msfusion, saved_runs/Pi3_ori_msfusion_highres,
saved_runs/Pi3_ori_ortho, saved_runs/Pi3_ori_ortho_highres,
saved_runs/Pi3_ori_ortho_sp_dr, saved_runs/Pi3_ori_ortho_sp_dr_highres,
saved_runs/S1, saved_runs/S2, and saved_runs/template.

The output directory is a live workspace rather than an immutable artifact store. Before
claiming a weight is the one used in the paper, record its modification time, config, and
the evaluation output that consumed it. auto_resume can make a rerun differ from a fresh
run even when the config name is unchanged.

## Reproducibility checklist

1. Use host 10.15.89.238, account wangqw, and the authoritative path above.
2. Record the current Git status before launching; the working tree already contains
   unrelated untracked Cross3R-dualsat files and configs.
3. Pass an explicit --config-name; do not rely on the launcher decorator's default.
4. Verify /data/zhongyao/dataset/ and /home/wangqw/NeurIPS26/dataset_lmdb_v2, including
   LMDB metadata keys.
5. Confirm the stage-1 checkpoint exists before launching the high-resolution config.
6. Record GPU assignment, log path, whether auto_resume restored a state, and the final
   best_model path.
7. Run evaluation from /home/wangqw/video_program/Pi3_eval/Pi3 using the paired
   AGENT.md and the exact evaluation config; do not infer metrics from a training log.

## Evidence anchors

- Entry/config: scripts/train_pi3.py:9-12, CLAUDE.md:11-19,25-55,
  configs/C1_no_dualcam.yaml, configs/C1_no_dualcam_highres.yaml.
- Model contract: configs/model/pi3_googlestreet_sat_C1_no_dualcam.yaml,
  pi3/models/pi3_training_sat.py, pi3/models/cross3r.py.
- Data contract: configs/data/googlestreet_sat.yaml,
  datasets/googlestreet_dataset.py, datasets/base/base_dataset.py.
- Optimization/checkpoint contract: configs/train/train_pi3_lowres_megadepth.yaml,
  configs/train/train_pi3_highres.yaml, trainers/base_trainer_accelerate.py,
  trainers/pi3_trainer.py.
- Experiment launchers: scripts/run_ablation_queue.sh, scripts/run_rawh_queue.sh,
  scripts/run_satinject_queue.sh.
