"""KITTI fine-tuning trainer for Pi3.

Subclass of ``Pi3Trainer`` that:
  * uses the KITTI-specific loss (``pi3.models.loss_kitti.Pi3KittiLoss``)
  * optionally freezes the point-cloud branch so only the camera branch
    learns from KITTI (no point GT → keeps C1_highres point quality intact)

Set ``cfg.train.freeze_point_branch = True`` (default False) to freeze
``point_decoder``, ``point_head``, and ``sat_register_inject``. When this
flag is on we additionally freeze the shared backbone components
(``encoder``, ``decoder``, ``register_token``, ``sat_pos_embedder``,
``ms_fusion``) to enforce that nothing upstream of the camera branch
moves either — anything else would silently corrupt the point branch's
inputs.

The camera branch (``camera_decoder``, ``camera_head``,
``camera_head_sat``, ``sat_register_inject_camera``) plus the sat scale
head (``sat_mpp_head``) stay trainable; the projection-consistency loss
gives gradient to ``sat_mpp_head`` so it can adapt KITTI's mpp.
"""
from trainers.pi3_trainer import Pi3Trainer


def _freeze_module(m):
    if m is None:
        return 0
    n = 0
    for p in m.parameters():
        if p.requires_grad:
            p.requires_grad = False
            n += 1
    return n


class Pi3KittiTrainer(Pi3Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        if bool(cfg.train.get('freeze_point_branch', False)):
            self._freeze_point_branch()

    def _freeze_point_branch(self):
        # After super().__init__ runs accelerator.prepare the model is
        # wrapped (DDP/FSDP/etc.); the original Pi3 lives at .module.
        model = self.model
        while hasattr(model, 'module'):
            model = model.module
        # Modules to freeze: point branch + shared backbone (everything
        # upstream of / unique to the point pipeline).
        frozen_modules = [
            ('encoder', getattr(model, 'encoder', None)),
            ('decoder', getattr(model, 'decoder', None)),
            ('register_token_param', None),  # handled separately below
            ('sat_pos_embedder', getattr(model, 'sat_pos_embedder', None)),
            ('ms_fusion', getattr(model, 'ms_fusion', None)),
            ('point_decoder', getattr(model, 'point_decoder', None)),
            ('point_head', getattr(model, 'point_head', None)),
            ('sat_register_inject', getattr(model, 'sat_register_inject', None)),
            ('sat_mpp_head', getattr(model, 'sat_mpp_head', None)),
        ]
        total = 0
        for name, mod in frozen_modules:
            if mod is None:
                continue
            n = _freeze_module(mod)
            if n:
                print(f'[Pi3KittiTrainer] Froze {name}: {n} params')
            total += n
        # register_token is an nn.Parameter, not a module
        if hasattr(model, 'register_token') and isinstance(
            model.register_token, type(model.register_token)
        ) and model.register_token.requires_grad:
            model.register_token.requires_grad = False
            total += 1
            print('[Pi3KittiTrainer] Froze register_token')

        # Sanity: list what stays trainable.
        trainable = [
            n for n, p in model.named_parameters() if p.requires_grad
        ]
        print(
            f'[Pi3KittiTrainer] freeze_point_branch=True → '
            f'froze {total} params; {len(trainable)} trainable params remain.'
        )
        # Print the top-level module owners of the trainable params (de-duped).
        roots = sorted({n.split('.')[0] for n in trainable})
        print(f'[Pi3KittiTrainer] Trainable module roots: {roots}')
