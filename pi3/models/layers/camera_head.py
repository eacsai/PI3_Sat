import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F

# code adapted from 'https://github.com/nianticlabs/marepo/blob/9a45e2bb07e5bb8cb997620088d352b439b13e0e/transformer/transformer.py#L172'
class ResConvBlock(nn.Module):
    """
    1x1 convolution residual block
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head_skip = nn.Identity() if self.in_channels == self.out_channels else nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        # self.res_conv1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        # self.res_conv2 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)
        # self.res_conv3 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)

        # change 1x1 convolution to linear
        self.res_conv1 = nn.Linear(self.in_channels, self.out_channels)
        self.res_conv2 = nn.Linear(self.out_channels, self.out_channels)
        self.res_conv3 = nn.Linear(self.out_channels, self.out_channels)

    def forward(self, res):
        x = F.relu(self.res_conv1(res))
        x = F.relu(self.res_conv2(x))
        x = F.relu(self.res_conv3(x))
        res = self.head_skip(res) + x
        return res

class CameraHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        output_dim = dim
        self.res_conv = nn.ModuleList([deepcopy(ResConvBlock(output_dim, output_dim)) 
                for _ in range(2)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.more_mlps = nn.Sequential(
            nn.Linear(output_dim,output_dim),
            nn.ReLU(),
            nn.Linear(output_dim,output_dim),
            nn.ReLU()
            )
        self.fc_t = nn.Linear(output_dim, 3)
        self.fc_rot = nn.Linear(output_dim, 9)

    def forward(self, feat, patch_h, patch_w):
        BN, hw, c = feat.shape

        for i in range(2):
            feat = self.res_conv[i](feat)

        # feat = self.avgpool(feat)
        feat = self.avgpool(feat.permute(0, 2, 1).reshape(BN, -1, patch_h, patch_w).contiguous())              ##########
        feat = feat.view(feat.size(0), -1)

        feat = self.more_mlps(feat)  # [B, D_]
        with torch.amp.autocast(device_type='cuda', enabled=False):
            out_t = self.fc_t(feat.float())  # [B,3]
            out_r = self.fc_rot(feat.float())  # [B,9]
            pose = self.convert_pose_to_4x4(BN, out_r, out_t, feat.device)

        return pose

    def convert_pose_to_4x4(self, B, out_r, out_t, device):
        out_r = self.svd_orthogonalize(out_r)  # [N,3,3]
        pose = torch.zeros((B, 4, 4), device=device)
        pose[:, :3, :3] = out_r
        pose[:, :3, 3] = out_t
        pose[:, 3, 3] = 1.
        return pose

    def svd_orthogonalize(self, m):
        """Convert 9D representation to SO(3) using SVD orthogonalization.

        Args:
          m: [BATCH, 3, 3] 3x3 matrices.

        Returns:
          [BATCH, 3, 3] SO(3) rotation matrices.
        """
        if m.dim() < 3:
            m = m.reshape((-1, 3, 3))
        m_transpose = torch.transpose(torch.nn.functional.normalize(m, p=2, dim=-1), dim0=-1, dim1=-2)
        u, s, v = torch.svd(m_transpose)
        det = torch.det(torch.matmul(v, u.transpose(-2, -1)))
        # Check orientation reflection.
        r = torch.matmul(
            torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2),
            u.transpose(-2, -1)
        )
        return r


class CameraHeadSatResidual(nn.Module):
    """C3b: full 6-DOF sat head, but predicts SMALL residual on top of learnable anchor pose.

    - anchor_R (shared, 6D parameterized) and anchor_t (shared, 3D) — single learnable pose for whole dataset
    - fc_rot output → small rotation residual (delta_R ≈ identity at init thanks to residual_scale)
    - fc_t output → small translation residual
    - Final: R = anchor_R @ delta_R, t = anchor_t + delta_t
    """
    def __init__(self, dim=512, residual_scale_t=0.1, residual_scale_r=0.1):
        super().__init__()
        self.res_conv = nn.ModuleList([deepcopy(ResConvBlock(dim, dim)) for _ in range(2)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.more_mlps = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
        )
        self.fc_t = nn.Linear(dim, 3)
        self.fc_rot = nn.Linear(dim, 9)
        # Shared learnable anchor pose (init = identity rotation, zero translation).
        self.anchor_R_6d = nn.Parameter(torch.tensor([1., 0., 0., 0., 1., 0.]))
        self.anchor_t = nn.Parameter(torch.zeros(3))
        self.residual_scale_t = residual_scale_t
        self.residual_scale_r = residual_scale_r

    @staticmethod
    def six_d_to_R(x6):
        a1, a2 = x6[..., :3], x6[..., 3:6]
        b1 = F.normalize(a1, dim=-1)
        b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-1)

    @staticmethod
    def svd_orthogonalize(m):
        if m.dim() < 3:
            m = m.reshape((-1, 3, 3))
        m_t = torch.transpose(F.normalize(m, p=2, dim=-1), dim0=-1, dim1=-2)
        u, s, v = torch.svd(m_t)
        det = torch.det(torch.matmul(v, u.transpose(-2, -1)))
        r = torch.matmul(
            torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2),
            u.transpose(-2, -1),
        )
        return r

    def forward(self, feat, patch_h, patch_w):
        BN, hw, c = feat.shape
        for blk in self.res_conv:
            feat = blk(feat)
        feat = self.avgpool(feat.permute(0, 2, 1).reshape(BN, -1, patch_h, patch_w).contiguous())
        feat = feat.view(BN, -1)
        feat = self.more_mlps(feat)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            feat_f = feat.float()
            # Residual translation (small)
            delta_t = self.fc_t(feat_f) * self.residual_scale_t           # (BN, 3)
            # Residual rotation: identity + small perturbation, then orthogonalize → near-identity rotation
            delta_R_raw = self.fc_rot(feat_f).view(BN, 3, 3) * self.residual_scale_r
            eye = torch.eye(3, device=feat_f.device, dtype=feat_f.dtype).expand(BN, 3, 3)
            delta_R = self.svd_orthogonalize(eye + delta_R_raw)             # (BN, 3, 3)
            # Compose with anchor
            anchor_R = self.six_d_to_R(self.anchor_R_6d).to(feat_f.dtype)   # (3, 3)
            R = anchor_R.unsqueeze(0) @ delta_R                             # (BN, 3, 3)
            t = self.anchor_t.unsqueeze(0).to(feat_f.dtype) + delta_t       # (BN, 3)
            pose = torch.zeros((BN, 4, 4), device=feat_f.device, dtype=feat_f.dtype)
            pose[:, :3, :3] = R
            pose[:, :3, 3] = t
            pose[:, 3, 3] = 1.
        return pose


class CameraHeadSatConstrained(nn.Module):
    """Sat camera with restricted DOF: only (x, y, z, yaw) per sample + shared learnable world rotation.

    Sat is always top-down → 5 of 6 rotation DOF are physically constant (modulo dataset convention).
    Total: 4 free per sample + 3 shared (world rotation correction, init=identity, model learns it).
    """
    def __init__(self, dim=512):
        super().__init__()
        self.res_conv = nn.ModuleList([deepcopy(ResConvBlock(dim, dim)) for _ in range(2)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.more_mlps = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
        )
        self.fc_t = nn.Linear(dim, 3)
        self.fc_yaw = nn.Linear(dim, 1)
        # Shared learnable world frame (top-down) rotation, parameterized as 6D for stable orthogonalization.
        # Init = first 2 columns of identity matrix → R_world starts as identity (model learns the actual top-down).
        init_6d = torch.tensor([1., 0., 0., 0., 1., 0.])  # col0 + col1 of I
        self.world_R_6d = nn.Parameter(init_6d)

    @staticmethod
    def six_d_to_R(x6):
        # x6: (..., 6) — first 3 = column 0, next 3 = column 1
        a1, a2 = x6[..., :3], x6[..., 3:6]
        b1 = F.normalize(a1, dim=-1)
        b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-1)  # (..., 3, 3)

    def forward(self, feat, patch_h, patch_w):
        BN, hw, c = feat.shape
        for blk in self.res_conv:
            feat = blk(feat)
        feat = self.avgpool(feat.permute(0, 2, 1).reshape(BN, -1, patch_h, patch_w).contiguous())
        feat = feat.view(BN, -1)
        feat = self.more_mlps(feat)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            feat_f = feat.float()
            t = self.fc_t(feat_f)                       # (BN, 3)
            yaw = self.fc_yaw(feat_f).squeeze(-1)        # (BN,)
            cy, sy = torch.cos(yaw), torch.sin(yaw)
            # R_yaw rotates around camera z-axis (which is the look-down axis after R_world)
            R_yaw = torch.zeros((BN, 3, 3), device=feat_f.device, dtype=feat_f.dtype)
            R_yaw[:, 0, 0] = cy;  R_yaw[:, 0, 1] = -sy
            R_yaw[:, 1, 0] = sy;  R_yaw[:, 1, 1] = cy
            R_yaw[:, 2, 2] = 1.0
            # Shared learnable world rotation
            R_world = self.six_d_to_R(self.world_R_6d).to(R_yaw.dtype)  # (3, 3)
            R = R_world.unsqueeze(0) @ R_yaw  # (BN, 3, 3)
            pose = torch.zeros((BN, 4, 4), device=feat_f.device, dtype=feat_f.dtype)
            pose[:, :3, :3] = R
            pose[:, :3, 3] = t
            pose[:, 3, 3] = 1.
        return pose