"""Cross3R-DualSat — sat branch supervises BOTH ortho and perspective heads.

Same architecture as Cross3R (`cross3r.py`), but the satellite views
emit two independent local-point predictions per pixel:

  - ortho_sat:  ``[sat_xy_base * sat_mpp, z]`` (current Cross3R behaviour)
  - persp_sat:  ``[xy * z, z]``               (same parameterisation as ground/UAV)

Both are returned in the forward dict and supervised at training time
with equal weight. At inference, ``sat_inference_mode`` selects which
parameterisation populates the primary ``local_points`` field that
downstream global-point projection / metric code reads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .dinov2.layers import Mlp
from .dinov2.hub.backbones import dinov2_vitl14_reg
from ..utils.geometry import homogenize_points
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d
from .layers.camera_head import CameraHead
from .sat_position import FourierEmbedder
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file


MIN_MPP = 0.005
MAX_MPP = 0.05


class MultiScaleFusion(nn.Module):
    """Softmax-gated fusion of features from multiple decoder layers."""

    def __init__(self, dim: int, num_layers: int = 4, init_layer_idx: int = -1):
        super().__init__()
        self.num_layers = num_layers
        init_w = torch.full((num_layers,), -5.0)
        init_w[init_layer_idx] = 5.0
        self.gate_logits = nn.Parameter(init_w)

    def forward(self, feats):
        assert len(feats) == self.num_layers
        weights = torch.softmax(self.gate_logits, dim=0)
        return sum(w * f for w, f in zip(weights, feats))


class SatRegisterInjection(nn.Module):
    """Broadcast a satellite global summary into ground/UAV view tokens."""

    def __init__(self, dim: int = 1024, num_register: int = 5):
        super().__init__()
        self.num_register = num_register
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden: torch.Tensor, is_sat_mask: torch.Tensor) -> torch.Tensor:
        BN, S, D = hidden.shape
        B, N = is_sat_mask.shape
        ph = hidden.view(B, N, S, D)
        sat_w = is_sat_mask.to(ph.dtype).view(B, N)
        sat_w_sum = sat_w.sum(dim=1, keepdim=True).clamp_min(1.0)
        per_view = ph[:, :, self.num_register:, :].mean(dim=2)
        sat_pooled = (per_view * sat_w.unsqueeze(-1)).sum(dim=1) / sat_w_sum
        injected = self.gate * self.ffn(self.norm(sat_pooled))
        ground_mask = (~is_sat_mask).to(ph.dtype).view(B, N, 1, 1)
        broadcast = injected.view(B, 1, 1, D) * ground_mask
        return (ph + broadcast).view(BN, S, D)


def _freeze(modules):
    for m in modules:
        try:
            for p in m.parameters():
                p.requires_grad = False
        except AttributeError:
            m.requires_grad = False


class Cross3RDualSat(nn.Module):
    """Cross3R variant whose sat branch outputs both ortho and perspective heads."""

    def __init__(
        self,
        load_pi3: bool = True,
        freeze_encoder: bool = True,
        num_dec_blk_not_to_checkpoint: int = 4,
        ckpt: str | None = None,
        sat_inference_mode: str = "ortho",
    ):
        super().__init__()

        assert sat_inference_mode in ("ortho", "persp"), sat_inference_mode
        self.sat_inference_mode = sat_inference_mode

        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token

        self.rope = RoPE2D(freq=100.0)
        self.position_getter = PositionGetter()

        self.sat_pos_embedder = FourierEmbedder(
            in_dim=2, embed_dim=self.encoder.embed_dim, num_freqs=64, scale=1.5,
        )

        dec_embed_dim = 1024
        dec_num_heads = 16
        dec_depth = 36
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=4,
                qkv_bias=True, proj_bias=True, ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU, ffn_layer=Mlp,
                init_values=0.01, qk_norm=True,
                attn_class=FlashAttentionRope, rope=self.rope,
            )
            for _ in range(dec_depth)
        ])
        self.dec_embed_dim = dec_embed_dim
        self.num_dec_blk_not_to_checkpoint = num_dec_blk_not_to_checkpoint

        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(
            torch.randn(1, 2, num_register_tokens, dec_embed_dim)
        )
        nn.init.normal_(self.register_token, std=1e-6)

        self.point_decoder = TransformerDecoder(
            in_dim=2 * dec_embed_dim,
            dec_embed_dim=1024, dec_num_heads=16, out_dim=1024,
            rope=self.rope,
        )
        self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        self.ms_collect_indices = (8, 17, 26, 34)
        self.ms_fusion = MultiScaleFusion(dim=dec_embed_dim, num_layers=4, init_layer_idx=-1)

        self.sat_register_inject = SatRegisterInjection(dim=1024, num_register=num_register_tokens)

        self.sat_mpp_head = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

        self.camera_decoder = TransformerDecoder(
            in_dim=2 * dec_embed_dim,
            dec_embed_dim=1024, dec_num_heads=16, out_dim=512,
            rope=self.rope, use_checkpoint=False,
        )
        self.camera_head = CameraHead(dim=512)

        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if ckpt is not None:
            state = torch.load(ckpt, weights_only=False, map_location='cpu')
            res = self.load_state_dict(state, strict=False)
            print(f'[Cross3RDualSat] Loaded checkpoint from {ckpt}: {res}')
            del state
            torch.cuda.empty_cache()
        elif load_pi3:
            pi3_weight = load_file('ckpts/Pi3/model_pi3.safetensors')
            if 'register_token' in pi3_weight:
                rt = pi3_weight['register_token']
                if rt.shape[1] == 1 and self.register_token.shape[1] == 2:
                    pi3_weight['register_token'] = rt.repeat(1, 2, 1, 1)
            res = self.load_state_dict(pi3_weight, strict=False)
            print(f'[Cross3RDualSat] Loaded π³ weights: {res}')

        if freeze_encoder:
            print('[Cross3RDualSat] Freezing the encoder.')
            _freeze([self.encoder])

    def decode(self, hidden, N, H, W, is_sat_mask, sat_pos_embed):
        BN, hw, _ = hidden.shape
        B = BN // N
        hidden = hidden.reshape(B * N, hw, -1)

        reg_sat = self.register_token[:, 0:1].expand(B, N, -1, -1)
        reg_non = self.register_token[:, 1:2].expand(B, N, -1, -1)
        mask = is_sat_mask.to(hidden.device).view(B, N, 1, 1)
        register_token = torch.where(mask, reg_sat, reg_non).reshape(
            B * N, *self.register_token.shape[-2:]
        )

        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.patch_start_idx > 0:
            pad = torch.zeros(
                B, self.patch_start_idx, sat_pos_embed.shape[-1],
                device=sat_pos_embed.device, dtype=sat_pos_embed.dtype,
            )
            sat_pos_embed = torch.cat([pad, sat_pos_embed], dim=1)
        hidden = hidden.reshape(B, N, hw, -1)
        sat_view_mask = is_sat_mask.to(hidden.device).unsqueeze(-1).unsqueeze(-1).float()
        hidden = hidden + sat_view_mask * sat_pos_embed[:, None, :, :]
        hidden = hidden.reshape(B * N, hw, -1)

        pos = self.position_getter(B * N, H // self.patch_size, W // self.patch_size, hidden.device)
        pos = pos + 1
        pos_special = torch.zeros(B * N, self.patch_start_idx, 2,
                                  device=hidden.device, dtype=pos.dtype)
        pos = torch.cat([pos_special, pos], dim=1)

        ms_collected = []
        last_hidden = None
        for i, blk in enumerate(self.decoder):
            if i % 2 == 0:
                pos = pos.reshape(B * N, hw, -1)
                hidden = hidden.reshape(B * N, hw, -1)
            else:
                pos = pos.reshape(B, N * hw, -1)
                hidden = hidden.reshape(B, N * hw, -1)

            if i >= self.num_dec_blk_not_to_checkpoint and self.training:
                hidden = checkpoint(blk, hidden, xpos=pos, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=pos)

            if i in self.ms_collect_indices:
                ms_collected.append(hidden.reshape(B * N, hw, -1))
            if i == len(self.decoder) - 1:
                last_hidden = hidden.reshape(B * N, hw, -1)

        fused = self.ms_fusion(ms_collected)
        return torch.cat([fused, last_hidden], dim=-1), pos.reshape(B * N, hw, -1)

    def forward(self, imgs, is_sat_mask):
        imgs = (imgs - self.image_mean) / self.image_std
        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14

        imgs = imgs.reshape(B * N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)
        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        y_steps = torch.linspace(-1, 1, patch_h, device=hidden.device, dtype=hidden.dtype)
        x_steps = torch.linspace(-1, 1, patch_w, device=hidden.device, dtype=hidden.dtype)
        grid_y, grid_x = torch.meshgrid(y_steps, x_steps, indexing='ij')
        pos_input = torch.stack([grid_x, grid_y], dim=-1)
        sat_pos_embed = self.sat_pos_embedder(pos_input.unsqueeze(0).expand(B, -1, -1, -1))
        sat_pos_embed = sat_pos_embed.reshape(B, patch_h * patch_w, -1)

        hidden, pos = self.decode(hidden, N, H, W, is_sat_mask, sat_pos_embed)

        point_hidden = self.point_decoder(hidden, xpos=pos)
        point_hidden = self.sat_register_inject(point_hidden, is_sat_mask)

        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            point_hidden = point_hidden.float()
            camera_hidden = camera_hidden.float()

            ret = self.point_head(
                [point_hidden[:, self.patch_start_idx:]], (H, W)
            ).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = F.softplus(z) + 1e-6
            persp_points_all = torch.cat([xy * z, z], dim=-1)

            global_mpp_logit = self.sat_mpp_head(
                point_hidden[:, self.patch_start_idx:].mean(dim=1)
            ).reshape(B, N, 1, 1, 1)
            sat_mpp = MIN_MPP + (MAX_MPP - MIN_MPP) * torch.sigmoid(global_mpp_logit)

            x_idx = torch.arange(W, device=z.device, dtype=sat_mpp.dtype)
            y_idx = torch.arange(H, device=z.device, dtype=sat_mpp.dtype)
            u = (x_idx + 0.5) / W
            v = (y_idx + 0.5) / H
            grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')
            sat_grid = torch.stack([grid_u, grid_v], dim=-1)
            wh = torch.tensor([W, H], device=z.device, dtype=sat_mpp.dtype)
            sat_xy_base = (sat_grid - 0.5) * wh
            sat_xy_final = sat_xy_base[None, None, :, :, :] * sat_mpp
            ortho_sat_points_all = torch.cat([sat_xy_final, z], dim=-1)

            view_mask = is_sat_mask.to(z.device).view(B, N, 1, 1, 1)
            local_points_ortho = torch.where(view_mask, ortho_sat_points_all, persp_points_all)
            local_points_persp = persp_points_all

            if self.sat_inference_mode == "ortho":
                local_points = local_points_ortho
            else:
                local_points = local_points_persp

            cam_feat = camera_hidden[:, self.patch_start_idx:]
            camera_poses = self.camera_head(cam_feat, patch_h, patch_w).reshape(B, N, 4, 4)

            points = torch.einsum(
                'bnij, bnhwj -> bnhwi',
                camera_poses, homogenize_points(local_points),
            )[..., :3]

        return dict(
            points=points,
            local_points=local_points,
            local_points_ortho=local_points_ortho,
            local_points_persp=local_points_persp,
            camera_poses=camera_poses,
            sat_mpp=sat_mpp,
        )
