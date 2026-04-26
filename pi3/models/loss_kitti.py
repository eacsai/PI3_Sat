"""KITTI cross-view fine-tuning loss for Pi3.

Wraps the original ``pi3.models.loss.CameraLoss`` so KITTI fine-tunes
get exactly the same camera-pose supervision Pi3 uses internally
(scale-aligned Huber translation + angular rotation), with the same
``camera_loss * 0.1`` mixing factor that ``Pi3Loss.forward`` applies.

Differences from the full ``Pi3Loss``:
  * No point-cloud term (KITTI has no depth GT).
  * The per-batch scale that ``CameraLoss`` needs is computed from the
    pose translations themselves (mean GT translation magnitude / mean
    pred translation magnitude) instead of from the GT point cloud's
    average distance, since we don't have a usable point cloud.

An optional sat-projection consistency term is also implemented (same
formula as ``mv_recon/eval_sat.py``) but is disabled by default
(``proj_weight=0``) — Pi3's ``sat_mpp_head`` is sigmoid-bounded to
[0.005, 0.05] m/px, far below KITTI's ~0.2 m/px, so the projection
loss is structurally broken on KITTI and only fights the pose loss.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pi3.utils.geometry import se3_inverse
from pi3.models.loss import CameraLoss


class Pi3KittiLoss(nn.Module):
    """Pi3 ``CameraLoss`` (×0.1) plus optional sat-projection consistency.

    Args:
        trans_alpha: ``CameraLoss.alpha`` (Huber-trans weight). Default
            100 to match the original Pi3 setting.
        camera_weight: outer multiplier on the camera-loss output.
            Default 0.1, matching ``Pi3Loss.forward``'s mixing weight.
        proj_weight: weight on the optional sat-projection loss.
            Default 0.0 (disabled).
        proj_huber_delta: Huber delta for the projection error,
            normalised by image side. Default 0.05.
    """

    def __init__(
        self,
        trans_alpha: float = 100.0,
        camera_weight: float = 0.1,
        proj_weight: float = 0.0,
        proj_huber_delta: float = 0.05,
        # accept-and-ignore so the same config schema as Pi3Loss works
        train_conf: bool = False,
        normal_loss_start_epoch: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.camera_loss = CameraLoss(alpha=float(trans_alpha))
        self.camera_weight = float(camera_weight)
        self.proj_weight = float(proj_weight)
        self.proj_huber_delta = float(proj_huber_delta)

    # --- helpers -------------------------------------------------------
    @staticmethod
    def _stack_view_field(views, key):
        return torch.stack([v[key] for v in views], dim=1)

    @staticmethod
    def _scale_from_translations(pred_pose: torch.Tensor,
                                 gt_pose: torch.Tensor) -> torch.Tensor:
        """Per-sample scale = mean(|t_gt_rel|) / mean(|t_pred_rel|).

        Matches what ``Pi3Loss.prepare_gt`` would derive from a GT point
        cloud's average distance, but is computed from pose translations
        directly (no point cloud needed).  Detached: scale is a
        normalisation factor, not something we backprop into.
        """
        B, N = pred_pose.shape[:2]
        pred_w2c = se3_inverse(pred_pose)
        gt_w2c = se3_inverse(gt_pose)
        pred_rel_t = torch.matmul(
            pred_w2c.unsqueeze(2), pred_pose.unsqueeze(1)
        )[..., :3, 3]
        gt_rel_t = torch.matmul(
            gt_w2c.unsqueeze(2), gt_pose.unsqueeze(1)
        )[..., :3, 3]
        mask = ~torch.eye(N, dtype=torch.bool, device=pred_pose.device)
        pred_norm = pred_rel_t[:, mask].norm(dim=-1).mean(dim=-1)
        gt_norm = gt_rel_t[:, mask].norm(dim=-1).mean(dim=-1)
        scale = gt_norm / pred_norm.clamp_min(1e-6)
        scale = torch.where(torch.isfinite(scale) & (pred_norm > 1e-4),
                            scale, torch.ones_like(scale))
        return scale.detach()

    def _projection_loss(self, pred_pose: torch.Tensor,
                         pred_sat_mpp: torch.Tensor,
                         is_sat_mask: torch.Tensor,
                         uv_gt: torch.Tensor,
                         data_h: int, data_w: int):
        B, N = pred_pose.shape[:2]
        if N != 2:
            raise NotImplementedError("KITTI loss assumes exactly 2 views.")
        sat_idx = is_sat_mask.float().argmax(dim=1)
        grd_idx = (~is_sat_mask).float().argmax(dim=1)
        bidx = torch.arange(B, device=pred_pose.device)

        pose_sat = pred_pose[bidx, sat_idx]
        pose_grd = pred_pose[bidx, grd_idx]
        T_sat = pose_sat[:, :3, 3]
        R_sat = pose_sat[:, :3, :3]
        T_grd = pose_grd[:, :3, 3]
        delta = T_grd - T_sat
        P_local = torch.einsum('bij,bj->bi', R_sat.transpose(-1, -2), delta)

        if pred_sat_mpp is None:
            zero = torch.zeros((), device=pred_pose.device, dtype=pred_pose.dtype)
            return zero, dict(proj_loss=zero, proj_pix_err=zero)
        if pred_sat_mpp.ndim >= 3:
            mpp_per_view = pred_sat_mpp.reshape(B, N, -1)[..., 0]
            mpp = mpp_per_view[bidx, sat_idx]
        elif pred_sat_mpp.ndim == 2:
            mpp = pred_sat_mpp[bidx, 0]
        else:
            mpp = pred_sat_mpp.expand(B)
        mpp = mpp.clamp_min(1e-6)
        u_pred = P_local[:, 0] / mpp + data_w / 2.0
        v_pred = P_local[:, 1] / mpp + data_h / 2.0
        uv_pred = torch.stack([u_pred, v_pred], dim=-1)
        uv_gt_grd = uv_gt[bidx, grd_idx]
        side = float(min(data_h, data_w))
        err = (uv_pred - uv_gt_grd) / side
        proj_loss = F.huber_loss(err, torch.zeros_like(err),
                                 reduction='mean', delta=self.proj_huber_delta)
        return proj_loss, dict(
            proj_loss=proj_loss,
            proj_pix_err=((uv_pred - uv_gt_grd).norm(dim=-1).mean()),
            pred_sat_mpp=mpp.mean(),
        )

    # --- top-level forward ---------------------------------------------
    def forward(self, pred: dict, gt: list, epoch: Optional[int] = None):
        pred_pose = pred['camera_poses']                       # (B, N, 4, 4)
        gt_pose = self._stack_view_field(gt, 'camera_pose')    # (B, N, 4, 4)
        is_sat_mask = self._stack_view_field(gt, 'is_satellite').bool()

        scale = self._scale_from_translations(pred_pose, gt_pose)

        # Use Pi3's CameraLoss exactly (scale-aligned Huber trans + angular rot).
        cam_pred = {'camera_poses': pred_pose}
        cam_gt = {'camera_poses': gt_pose}
        camera_loss, cam_details = self.camera_loss(cam_pred, cam_gt, scale)

        total = self.camera_weight * camera_loss
        details = dict(
            **cam_details,
            scale=scale.mean(),
            camera_loss_pre_weight=camera_loss.detach(),
        )

        if self.proj_weight > 0.0:
            _, _, _, H, W = self._stack_view_field(gt, 'img').shape
            uv_gt = self._stack_view_field(gt, 'kitti_ground_uv_in_sat')
            proj_loss, proj_details = self._projection_loss(
                pred_pose=pred_pose,
                pred_sat_mpp=pred.get('sat_mpp', None),
                is_sat_mask=is_sat_mask,
                uv_gt=uv_gt,
                data_h=int(H), data_w=int(W),
            )
            total = total + self.proj_weight * proj_loss
            details.update(proj_details)

        return total, details
