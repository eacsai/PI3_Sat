"""Loss for Cross3R-DualSat: supervises both ortho and perspective sat heads.

Wraps the standard ``Pi3Loss`` (from ``loss.py``) and adds an auxiliary
point-loss term computed against ``pred['local_points_persp']``. The
two terms have equal weight; the camera loss runs once on the
ortho-normalised camera_poses (the persp head doesn't change camera
predictions, only the local-points parameterisation).
"""

from typing import Optional

import torch

from .loss import Pi3Loss


class Pi3LossDualSat(Pi3Loss):
    def forward(self, pred, gt_raw, epoch: Optional[int] = None):
        local_persp = pred.get('local_points_persp')
        local_ortho = pred.get('local_points_ortho')

        if local_ortho is not None:
            pred['local_points'] = local_ortho

        loss, details = super().forward(pred, gt_raw, epoch)

        if local_persp is not None and local_ortho is not None:
            gt_normalized = self.prepare_gt(gt_raw)
            masks = gt_normalized['valid_masks']
            B, N, H, W, _ = local_persp.shape
            tmp = local_persp.clone()
            tmp[~masks] = 0
            tmp = tmp.reshape(B, N, -1, 3)
            all_dis = tmp.norm(dim=-1)
            norm_b = all_dis.sum(dim=[-1, -2]) / (masks.float().sum(dim=[-1, -2, -3]) + 1e-8)
            local_persp_norm = local_persp / norm_b[..., None, None, None, None]

            pred_aux = {'local_points': local_persp_norm}
            point_loss_aux, details_aux, _ = self.point_loss(
                pred_aux, gt_normalized, epoch=epoch,
            )
            loss = loss + point_loss_aux
            details['local_pts_loss_persp'] = details_aux.get(
                'local_pts_loss', point_loss_aux,
            )

        return loss, details
