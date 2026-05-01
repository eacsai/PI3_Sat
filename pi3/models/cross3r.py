"""Cross3R — π³ specialised for cross-altitude reconstruction (CrossGeo).

Architecture: π³-Large backbone (DINOv2 ViT-L encoder + 36-layer decoder)
augmented with five satellite-aware modules and a softplus depth activation.

The five additions over vanilla π³:
    1. MultiScaleFusion        — softmax-gated fusion of decoder layers 8/17/26/34
    2. SatRegisterInjection    — sat patch summary broadcast to ground/UAV views
    3. sat_pos_embedder        — Fourier positional encoding for sat patches
    4. doubled register tokens — 2 banks (sat / non-sat) instead of 1
    5. orthographic sat branch — sat_xy_base * sat_mpp metric-scale projection

Compared to the older C1_highres definition, the published Cross3R drops the
sat-specific camera head (`use_dual_camera_head=False`); it uses a single
shared camera head matching π³'s design, which improved every metric except
AUC@30 in the leave-one-out study.
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


# Sat meter-per-pixel is sigmoid-bounded to a physically plausible range.
MIN_MPP = 0.005
MAX_MPP = 0.05


class MultiScaleFusion(nn.Module):
    """Softmax-gated fusion of features from multiple decoder layers.

    Initial gate logits are biased so that the network starts as
    ``last-layer-only`` (an exact identity replacement of the standard
    π³ behaviour); during training the gate learns to weight earlier
    layers when their multi-scale information helps.
    """

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
    """Broadcast a satellite global summary into ground/UAV view tokens.

    The summary is the per-view mean of patch tokens (excluding register
    tokens), aggregated only over satellite views. The result is passed
    through a LayerNorm + 2-layer FFN (last layer zero-initialised) and
    a learnable scalar gate (also zero-initialised). The double soft-start
    means iteration-0 behaviour is identical to vanilla π³; the network
    only opts in when satellite information actually helps.
    """

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
        per_view = ph[:, :, self.num_register:, :].mean(dim=2)          # (B, N, D)
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


class Cross3R(nn.Module):
    """Cross3R: π³ backbone + 5 satellite-aware modules + softplus depth.

    Args:
        load_pi3: Initialise the shared layers from the public π³-Large
            checkpoint at ``ckpts/Pi3/model_pi3.safetensors``. The new
            modules are randomly initialised; their gates / last linear
            layers are zero-initialised so the model starts as a
            near-identity perturbation of π³.
        freeze_encoder: Freeze the DINOv2 encoder (recommended for
            fine-tuning on CrossGeo).
        num_dec_blk_not_to_checkpoint: Number of leading decoder blocks
            that skip gradient checkpointing (memory / speed trade-off).
        ckpt: Optional path to a Cross3R checkpoint to load (overrides
            ``load_pi3``).
    """

    def __init__(
        self,
        load_pi3: bool = True,
        freeze_encoder: bool = True,
        num_dec_blk_not_to_checkpoint: int = 4,
        ckpt: str | None = None,
    ):
        super().__init__()

        # ---- Encoder ----
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token

        # ---- Positional encoding (RoPE2D, freq=100) ----
        self.rope = RoPE2D(freq=100.0)
        self.position_getter = PositionGetter()

        # ---- Sat-specific Fourier positional encoding ----
        self.sat_pos_embedder = FourierEmbedder(
            in_dim=2, embed_dim=self.encoder.embed_dim, num_freqs=64, scale=1.5,
        )

        # ---- Decoder (36 layers, dim=1024, 16 heads) ----
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

        # ---- Doubled register tokens (sat bank + non-sat bank) ----
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(
            torch.randn(1, 2, num_register_tokens, dec_embed_dim)
        )
        nn.init.normal_(self.register_token, std=1e-6)

        # ---- Local-points head (shared by sat / grd / UAV) ----
        self.point_decoder = TransformerDecoder(
            in_dim=2 * dec_embed_dim,
            dec_embed_dim=1024, dec_num_heads=16, out_dim=1024,
            rope=self.rope,
        )
        self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # ---- MultiScaleFusion across decoder layers 8 / 17 / 26 / 34 ----
        self.ms_collect_indices = (8, 17, 26, 34)
        self.ms_fusion = MultiScaleFusion(dim=dec_embed_dim, num_layers=4, init_layer_idx=-1)

        # ---- SatRegisterInjection (sat→grd information flow) ----
        self.sat_register_inject = SatRegisterInjection(dim=1024, num_register=num_register_tokens)

        # ---- sat_mpp head: regress per-view meter-per-pixel scalar ----
        self.sat_mpp_head = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

        # ---- Camera-pose head (single shared head for all views) ----
        self.camera_decoder = TransformerDecoder(
            in_dim=2 * dec_embed_dim,
            dec_embed_dim=1024, dec_num_heads=16, out_dim=512,
            rope=self.rope, use_checkpoint=False,
        )
        self.camera_head = CameraHead(dim=512)

        # ---- Image normalisation buffers (ImageNet mean/std) ----
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # ---- Weight initialisation ----
        if ckpt is not None:
            state = torch.load(ckpt, weights_only=False, map_location='cpu')
            res = self.load_state_dict(state, strict=False)
            print(f'[Cross3R] Loaded checkpoint from {ckpt}: {res}')
            del state
            torch.cuda.empty_cache()
        elif load_pi3:
            pi3_weight = load_file('ckpts/Pi3/model_pi3.safetensors')
            # π³'s register_token is (1, 1, tokens, dim); we need (1, 2, tokens, dim)
            if 'register_token' in pi3_weight:
                rt = pi3_weight['register_token']
                if rt.shape[1] == 1 and self.register_token.shape[1] == 2:
                    pi3_weight['register_token'] = rt.repeat(1, 2, 1, 1)
            res = self.load_state_dict(pi3_weight, strict=False)
            print(f'[Cross3R] Loaded π³ weights: {res}')

        if freeze_encoder:
            print('[Cross3R] Freezing the encoder.')
            _freeze([self.encoder])

    # ------------------------------------------------------------------
    # Decoder pass with sat positional encoding + register-token routing
    # ------------------------------------------------------------------
    def decode(self, hidden, N, H, W, is_sat_mask, sat_pos_embed):
        BN, hw, _ = hidden.shape
        B = BN // N
        hidden = hidden.reshape(B * N, hw, -1)

        # Choose register tokens per view: bank 0 for sat, bank 1 for ground/UAV.
        reg_sat = self.register_token[:, 0:1].expand(B, N, -1, -1)
        reg_non = self.register_token[:, 1:2].expand(B, N, -1, -1)
        mask = is_sat_mask.to(hidden.device).view(B, N, 1, 1)
        register_token = torch.where(mask, reg_sat, reg_non).reshape(
            B * N, *self.register_token.shape[-2:]
        )

        # Concatenate register tokens with patch tokens.
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        # Inject sat positional encoding into satellite views only.
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

        # Build RoPE positions; register tokens get pos = 0 (no positional info).
        pos = self.position_getter(B * N, H // self.patch_size, W // self.patch_size, hidden.device)
        pos = pos + 1
        pos_special = torch.zeros(B * N, self.patch_start_idx, 2,
                                  device=hidden.device, dtype=pos.dtype)
        pos = torch.cat([pos_special, pos], dim=1)

        # Decoder loop with multi-scale collection at layers 8/17/26/34.
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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, imgs, is_sat_mask):
        """Args:
            imgs: (B, N, 3, H, W) RGB images in [0, 1].
            is_sat_mask: (B, N) bool, True for satellite views.
        """
        imgs = (imgs - self.image_mean) / self.image_std
        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14

        # DINOv2 encoding.
        imgs = imgs.reshape(B * N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)
        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        # Build sat positional encoding (Fourier features over normalised pixel grid).
        y_steps = torch.linspace(-1, 1, patch_h, device=hidden.device, dtype=hidden.dtype)
        x_steps = torch.linspace(-1, 1, patch_w, device=hidden.device, dtype=hidden.dtype)
        grid_y, grid_x = torch.meshgrid(y_steps, x_steps, indexing='ij')
        pos_input = torch.stack([grid_x, grid_y], dim=-1)
        sat_pos_embed = self.sat_pos_embedder(pos_input.unsqueeze(0).expand(B, -1, -1, -1))
        sat_pos_embed = sat_pos_embed.reshape(B, patch_h * patch_w, -1)

        # Decode with sat-aware register routing + sat positional encoding.
        hidden, pos = self.decode(hidden, N, H, W, is_sat_mask, sat_pos_embed)

        # Point branch: decoder + sat→grd register injection + linear point head.
        point_hidden = self.point_decoder(hidden, xpos=pos)
        point_hidden = self.sat_register_inject(point_hidden, is_sat_mask)

        # Camera branch (single shared head for all views).
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            point_hidden = point_hidden.float()
            camera_hidden = camera_hidden.float()

            # Ground/UAV: perspective back-projection [xy * z, z] with softplus depth.
            ret = self.point_head(
                [point_hidden[:, self.patch_start_idx:]], (H, W)
            ).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = F.softplus(z) + 1e-6
            grd_points_all = torch.cat([xy * z, z], dim=-1)

            # Satellite: orthographic back-projection [sat_xy_base * sat_mpp, z].
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
            sat_points_all = torch.cat([sat_xy_final, z], dim=-1)

            # Compose local points: sat views use orthographic, others use perspective.
            view_mask = is_sat_mask.to(z.device).view(B, N, 1, 1, 1)
            local_points = torch.where(view_mask, sat_points_all, grd_points_all)

            # Camera pose head (single shared head).
            cam_feat = camera_hidden[:, self.patch_start_idx:]
            camera_poses = self.camera_head(cam_feat, patch_h, patch_w).reshape(B, N, 4, 4)

            # Unproject local points using camera poses to obtain global points.
            points = torch.einsum(
                'bnij, bnhwj -> bnhwi',
                camera_poses, homogenize_points(local_points),
            )[..., :3]

        return dict(
            points=points,
            local_points=local_points,
            camera_poses=camera_poses,
            sat_mpp=sat_mpp,
        )
