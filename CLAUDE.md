# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Training code for **π³ (Pi3)**, a multi-view 3D reconstruction model. The upstream README describes three sequential training stages (low-res → high-res → confidence branch). This fork adds Google Street View + satellite ("sat") dataset variants and associated model/loss code.

## Running training

Training is launched via `accelerate` + Hydra. The Hydra entry point is `scripts/train_pi3.py`, which currently hardcodes `config_name="googlestreet_sat"` (see `scripts/train_pi3.py:9`) — override via the CLI or edit this line when switching experiments.

```bash
accelerate launch --config_file configs/accelerate/ddp.yaml \
  --num_processes 8 --num_machines 1 \
  scripts/train_pi3.py train=train_pi3_lowres name=pi3_lowres
```

Stage 2 / stage 3 from the README pass `model.ckpt=<path>` to chain checkpoints. Stage 3 requires `ckpts/segformer.b0.512x512.ade.160k.pth` to exist before launch.

To reduce GPU memory, in priority order: `train.max_img_per_gpu` ↓, `train.pixel_count_range` ↓, `model.num_dec_blk_not_to_checkpoint` ↑.

Data packing utility: `scripts/pack_googlestreet_lmdb.py` (builds an LMDB from the googlestreet dataset).

## Configuration layout (Hydra)

`configs/default.yaml` is the stock composition; `configs/googlestreet_sat.yaml`, `configs/megadepth.yaml`, etc. are top-level experiment configs that select a matching `model/`, `train/`, `data/`, `general/` group. When adding a new experiment, create one file per group and a new top-level composition file — do not edit `default.yaml`.

- `configs/train/train_pi3_lowres*.yaml` / `train_pi3_highres.yaml` / `train_pi3_conf.yaml` set resolution, image count range, batch caps, and the `trainer:` class path (e.g. `trainers.pi3_trainer.Pi3Trainer`).
- `configs/model/` controls architecture + checkpoint init. `model.ckpt: null` means train from scratch.
- `configs/data/` points each dataset loader at its `data_root` — **must be edited locally** before training.

## Code architecture

Entry flow: `scripts/train_pi3.py` → `hydra.main` loads config → `eval(cfg.trainer)(cfg).train()`. The `eval()` resolves the dotted path from config (e.g. `trainers.pi3_trainer.Pi3Trainer`), so `trainers/__init__.py` must export any new trainer.

**Trainers** (`trainers/`):
- `base_trainer_accelerate.BaseTrainer` — the Accelerate-based training loop, optimizer/scheduler build, checkpointing, logging.
- `pi3_trainer.Pi3Trainer` — Pi3-specific subclass. Splits parameters into `encoder.*` vs. the rest and applies separate LRs (`cfg.optimizer.encoder_lr` vs. `lr`) with weight-decay exclusion for 1-D params and biases. `before_epoch` propagates `set_epoch` through several layers of Accelerate-wrapped dataloader/sampler/batch_sampler (the deep `hasattr` chains are load-bearing — Accelerate wraps the sampler). When `cfg.train.random_reslution` is set, it resamples resolutions per epoch via `sample_resolutions` and pushes them into every underlying dataset via `_set_resolutions`.

**Model** (`pi3/models/`):
- `pi3.py` is the base architecture; `pi3_training*.py` files are task-specific training wrappers (`pi3_training_sat.py`, `pi3_training_query*.py`, …). Multiple `*_back.py` and `copy.py` files are experiment snapshots — treat them as scratch, not canonical.
- `loss.py` / `loss_sat.py` / `loss_query.py` are instantiated from config via `hydra.utils.instantiate(cfg.loss.train_loss)` in `Pi3Trainer.__init__`.
- `dinov2/`, `segformer/`, `layers/` hold backbone + block implementations.

**Datasets** (`datasets/`):
- Each dataset has its own module (`megadepth_dataset.py`, `googlestreet_dataset.py`, `megadepthsat_dataset.py`, `co3dv2_dataset.py`, `scannet_dataset.py`, `tartanair_dataset.py`). `googlestreet_dataset_back.py`, `*_new.py`, `copy` variants are scratch.
- `datasets/base/base_dataset.py` defines `sample_resolutions` and the `_set_resolutions` contract used by `Pi3Trainer.before_epoch`.
- Sampling strategy per the README: interval sampling with occasional consecutive frames; small scenes use random-from-sequence, large/long scenes use a dynamic sliding window with occasional sub-interval or whole-scene sampling.

## Gotchas

- `scripts/train_pi3.py` pins `config_name="googlestreet_sat"`. The README examples assume the stock `default` config — pass `--config-name=default` or edit the decorator when following README commands verbatim.
- `trainer:` in the config is resolved with `eval()`, so any new trainer class must be importable via its full dotted path and re-exported through `trainers/__init__.py`.
- `data_root` paths in `datasets/*.py` are hardcoded per machine and must be set before training will work.
- The `pi3/models/` and `datasets/` directories contain many `*_back.py`, `* copy.py`, `*_new.py` files — these are experimental snapshots, not the active code path. Check which module the current config actually imports before editing.
