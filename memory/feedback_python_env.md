---
name: use pi3 conda env for python
description: Always use /home/wangqw/.conda/envs/pi3/bin/python for all Python commands in the Pi3 project directory
type: feedback
originSessionId: c5e04f70-d3bd-48f2-aaee-ace18ff45f4c
---
All Python commands in the Pi3 project must use the pi3 conda environment.

**Why:** The system default Python doesn't have required packages (e.g., open3d). The pi3 env has everything needed for both training and evaluation.

**How to apply:** When running any Python script (training, eval, utilities), use `/home/wangqw/.conda/envs/pi3/bin/python` or activate the pi3 env first. This applies to both the training repo and the Pi3_eval repo.
