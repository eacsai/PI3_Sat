---
name: post-DDP module freeze must access .module
description: After accelerator.prepare wraps the model, `self.model.encoder` resolves to a DDP/FSDP wrapper attribute — to freeze child params you must walk through `.module`
type: feedback
originSessionId: 2593d9fb-88bf-4bdb-9677-2c7041f59fad
---
`Pi3Trainer.__init__` calls `super().__init__(cfg)` → `BaseTrainer.__init__`
runs `accelerator.prepare(model, optimizer, ...)` which wraps the model in
DDP / FSDP / etc. After that point, the original Pi3 instance lives at
`self.model.module` (or deeper for nested wrappers).

**Why:** When I first added `Pi3KittiTrainer._freeze_point_branch` it ran
`getattr(self.model, 'encoder', None)` after `super().__init__` returned.
DDP's wrapper has its own `encoder` attribute (the wrapped module's
forward, not the underlying nn.Module), so my freeze loop set
`requires_grad=False` on… nothing. Log showed `froze 0 params; 1178
trainable params remain`. Took a second restart to spot.

**How to apply:**
- Any `Pi3*Trainer` subclass that touches `requires_grad` post-init must
  walk `.module`:
  ```python
  model = self.model
  while hasattr(model, 'module'):
      model = model.module
  ```
  Then access `model.encoder`, `model.decoder`, etc.
- Alternative (cleaner): override `build_model` to freeze before
  `accelerator.prepare`. The optimizer will then exclude frozen params,
  saving optimiser-state memory. Post-prepare freezing wastes optim state
  but is functionally fine because frozen params have `grad=None` so
  AdamW skips their update.
- Sanity check: after freezing, log
  `[n for n, p in model.named_parameters() if p.requires_grad]` and
  inspect the top-level module roots to confirm only the intended
  submodules remain trainable.
