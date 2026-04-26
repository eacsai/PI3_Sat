"""Pi3 with frozen backbone + trainable satellite adapters.

All original Pi3 components are loaded from pretrained weights and frozen.
Only satellite-specific modules are trainable. When is_sat_mask has no
satellite views, the output is IDENTICAL to original Pi3.

Supports three modes via constructor flags:
  - use_sat_inject: SatPatchInjection after point_decoder (Exp16)
  - use_sat_cross_attn: SpatialSatCrossAttention after point_decoder (Exp17)
  - use_decoder_adapters: cross-attention adapters inside decoder (Exp18)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from copy import deepcopy

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d
from .layers.camera_head import CameraHead
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file
from .sat_position import FourierEmbedder


# ===================== Satellite Adapter Modules =====================

class SatPatchInjection(nn.Module):
    """Inject satellite patch-token summary into ground/drone tokens."""

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

    def forward(self, hidden, is_sat_mask):
        BN, S, D = hidden.shape
        B, N = is_sat_mask.shape
        ph = hidden.view(B, N, S, D)
        sat_w = is_sat_mask.to(ph.dtype).view(B, N)
        sat_w_sum = sat_w.sum(dim=1, keepdim=True).clamp_min(1.0)
        per_view_reg = ph[:, :, self.num_register:, :].mean(dim=2)
        sat_pooled = (per_view_reg * sat_w.unsqueeze(-1)).sum(dim=1) / sat_w_sum
        injected = self.ffn(self.norm(sat_pooled))
        injection = self.gate * injected
        ground_mask_f = (~is_sat_mask).to(ph.dtype).view(B, N, 1, 1)
        broadcast = injection.view(B, 1, 1, D) * ground_mask_f
        return (ph + broadcast).view(BN, S, D)


class SpatialSatCrossAttention(nn.Module):
    """Ground tokens cross-attend to satellite patch tokens."""

    def __init__(self, dim: int = 1024, hidden_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, hidden_dim)
        self.k_proj = nn.Linear(dim, hidden_dim)
        self.v_proj = nn.Linear(dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden, is_sat_mask):
        BN, S, D = hidden.shape
        B, N = is_sat_mask.shape
        ph = hidden.view(B, N, S, D)
        sat_w = is_sat_mask.to(ph.dtype)
        sat_w_sum = sat_w.sum(dim=1, keepdim=True).clamp_min(1.0)
        sat_w_norm = (sat_w / sat_w_sum).unsqueeze(-1).unsqueeze(-1)
        sat_tokens = (ph * sat_w_norm).sum(dim=1)  # (B, S, D)
        ground_mask_f = (~is_sat_mask).to(ph.dtype)

        Q = self.q_proj(self.norm_q(ph.reshape(BN, S, D)))
        kv_input = self.norm_kv(sat_tokens)
        K = self.k_proj(kv_input).unsqueeze(1).expand(B, N, S, self.hidden_dim).reshape(BN, S, self.hidden_dim)
        V = self.v_proj(kv_input).unsqueeze(1).expand(B, N, S, self.hidden_dim).reshape(BN, S, self.hidden_dim)

        Q = Q.view(BN, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(BN, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(BN, S, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(Q, K, V)
        attn_out = attn_out.transpose(1, 2).reshape(BN, S, self.hidden_dim)

        delta = self.gate * self.out_proj(attn_out)
        delta = delta.view(B, N, S, D) * ground_mask_f.view(B, N, 1, 1)
        return (ph + delta).view(BN, S, D)


class _GroundToSatCrossAttention(nn.Module):
    """Mirror of SpatialSatCrossAttention but reversed direction.

    Ground/drone tokens (K/V) feed satellite tokens (Q target). Only sat
    views receive the delta. Zero-init so initial = no-op.
    """

    def __init__(self, dim: int = 1024, hidden_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, hidden_dim)
        self.k_proj = nn.Linear(dim, hidden_dim)
        self.v_proj = nn.Linear(dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden, is_sat_mask):
        BN, S, D = hidden.shape
        B, N = is_sat_mask.shape
        ph = hidden.view(B, N, S, D)
        grd_w = (~is_sat_mask).to(ph.dtype)
        grd_w_sum = grd_w.sum(dim=1, keepdim=True).clamp_min(1.0)
        grd_w_norm = (grd_w / grd_w_sum).unsqueeze(-1).unsqueeze(-1)
        grd_tokens = (ph * grd_w_norm).sum(dim=1)  # (B, S, D)
        sat_mask_f = is_sat_mask.to(ph.dtype)

        Q = self.q_proj(self.norm_q(ph.reshape(BN, S, D)))
        kv_input = self.norm_kv(grd_tokens)
        K = self.k_proj(kv_input).unsqueeze(1).expand(B, N, S, self.hidden_dim).reshape(BN, S, self.hidden_dim)
        V = self.v_proj(kv_input).unsqueeze(1).expand(B, N, S, self.hidden_dim).reshape(BN, S, self.hidden_dim)

        Q = Q.view(BN, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(BN, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(BN, S, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(Q, K, V)
        attn_out = attn_out.transpose(1, 2).reshape(BN, S, self.hidden_dim)

        delta = self.gate * self.out_proj(attn_out)
        delta = delta.view(B, N, S, D) * sat_mask_f.view(B, N, 1, 1)  # only sat views modified
        return (ph + delta).view(BN, S, D)


class BidirectionalSatCrossAttention(nn.Module):
    """Exp19: bidirectional sat <-> ground cross-attention.

    Sat→Ground path: ground/drone tokens cross-attend to sat (existing semantics).
    Ground→Sat path: sat tokens cross-attend to ground/drone (new direction).
    Both paths zero-init + independent gate so initial behavior is no-op.
    """

    def __init__(self, dim: int = 1024, hidden_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.sat_to_grd = SpatialSatCrossAttention(dim=dim, hidden_dim=hidden_dim, num_heads=num_heads)
        self.grd_to_sat = _GroundToSatCrossAttention(dim=dim, hidden_dim=hidden_dim, num_heads=num_heads)

    def forward(self, hidden, is_sat_mask):
        hidden = self.sat_to_grd(hidden, is_sat_mask)
        hidden = self.grd_to_sat(hidden, is_sat_mask)
        return hidden


class SatCameraAdapter(nn.Module):
    """Exp21: lightweight adapter on camera_hidden for sat views only.

    Sits between frozen camera_decoder and frozen camera_head. Remaps
    sat view camera_hidden closer to the distribution camera_head expects,
    preserving both frozen modules. Zero-init + gate so initial = no-op.
    """

    def __init__(self, dim: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, camera_hidden, is_sat_mask):
        BN, S, D = camera_hidden.shape
        B, N = is_sat_mask.shape
        h = camera_hidden.view(B, N, S, D)
        delta = self.gate * self.ffn(self.norm(camera_hidden))
        sat_mask = is_sat_mask.to(h.dtype).view(B, N, 1, 1)
        delta = delta.view(B, N, S, D) * sat_mask
        return (h + delta).view(BN, S, D)


class DecoderSatAdapter(nn.Module):
    """Lightweight cross-attention adapter for decoder layers.

    Inserted after a frozen decoder block. Ground/drone tokens (query)
    attend to satellite tokens (key/value). Zero-init so initial = no-op.
    """

    def __init__(self, dim: int = 1024, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden, is_sat_mask):
        """hidden: (B*N, S, D), is_sat_mask: (B, N)"""
        BN, S, D = hidden.shape
        B, N = is_sat_mask.shape
        h = hidden.view(B, N, S, D)

        # Extract sat tokens (keep spatial dim)
        sat_w = is_sat_mask.to(h.dtype)
        sat_w_sum = sat_w.sum(dim=1, keepdim=True).clamp_min(1.0)
        sat_w_norm = (sat_w / sat_w_sum).unsqueeze(-1).unsqueeze(-1)
        sat_feat = (h * sat_w_norm).sum(dim=1)  # (B, S, D)

        Q = self.q_proj(self.norm_q(hidden))  # (BN, S, D)
        kv = self.norm_kv(sat_feat)
        K = self.k_proj(kv).unsqueeze(1).expand(B, N, S, D).reshape(BN, S, D)
        V = self.v_proj(kv).unsqueeze(1).expand(B, N, S, D).reshape(BN, S, D)

        Q = Q.view(BN, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(BN, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(BN, S, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(Q, K, V)
        attn_out = attn_out.transpose(1, 2).reshape(BN, S, D)

        delta = self.gate * self.out_proj(attn_out)
        # Only apply to ground/drone views
        ground_mask = (~is_sat_mask).to(h.dtype).view(B, N, 1, 1)
        delta = delta.view(B, N, S, D) * ground_mask
        return (h + delta).view(BN, S, D)


# ===================== Frozen-backbone helper =====================

def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            module.requires_grad = False


# ===================== Main Model =====================

class Pi3(nn.Module):
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
            load_pi3=True,
            freeze_encoder=True,
            train_conf=False,
            num_dec_blk_not_to_checkpoint=4,
            ckpt=None,
            # --- Satellite adapter config ---
            use_sat_inject=True,
            use_sat_cross_attn=False,
            use_decoder_adapters=False,
            adapter_layers=(12, 24, 34),
            # Exp19: bidirectional sat <-> ground cross-attn (post point_decoder)
            use_bidirectional_cross_attn=False,
            # Exp20: parallel trainable sat decoder (per-decoder-layer split)
            use_sat_parallel_decoder=False,
            # Exp21: lightweight adapter on camera_hidden for sat views (after frozen camera_decoder)
            use_sat_camera_adapter=False,
            # Exp22: parallel trainable camera decoder for sat views (mirrors frozen camera_decoder)
            use_sat_parallel_camera=False,
            # Exp23: full parallel sat camera branch — separate decoder AND head, no gate, direct prediction
            use_sat_camera_branch=False,
            # Exp24: full parallel sat point branch — separate point_decoder + point_head, no gate, direct prediction
            use_sat_point_branch=False,
        ):
        super().__init__()

        # ===================== Original Pi3 Components =====================

        # Encoder
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token

        # Positional Encoding
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope = None
        if self.pos_type.startswith('rope'):
            if RoPE2D is None:
                raise ImportError("Cannot find cuRoPE2D")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError

        # Decoder
        if decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError(f"Only 'large' decoder supported, got {decoder_size}")

        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim, num_heads=dec_num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, proj_bias=True, ffn_bias=True, drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU, ffn_layer=Mlp, init_values=0.01,
                qk_norm=True, attn_class=FlashAttentionRope, rope=self.rope
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        # Register tokens — ground/drone (frozen after loading)
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token_grd = nn.Parameter(
            torch.randn(1, 1, num_register_tokens, dec_embed_dim))
        nn.init.normal_(self.register_token_grd, std=1e-6)

        # Point decoder & head
        self.point_decoder = TransformerDecoder(
            in_dim=2 * dec_embed_dim, dec_embed_dim=1024,
            dec_num_heads=16, out_dim=1024, rope=self.rope)
        self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # Camera decoder & head
        self.camera_decoder = TransformerDecoder(
            in_dim=2 * dec_embed_dim, dec_embed_dim=1024,
            dec_num_heads=16, out_dim=512, rope=self.rope, use_checkpoint=False)
        self.camera_head = CameraHead(dim=512)

        # ImageNet normalization
        self.register_buffer("image_mean",
                             torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std",
                             torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.train_conf = train_conf
        if train_conf:
            self.conf_decoder = deepcopy(self.point_decoder)
            self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)
        else:
            self.conf_decoder = None
            self.conf_head = None

        self.num_dec_blk_not_to_checkpoint = num_dec_blk_not_to_checkpoint

        # ===================== Trainable Satellite Adapters =====================

        # Satellite register tokens (trainable, separate from frozen ground/drone ones)
        self.register_token_sat = nn.Parameter(
            torch.randn(1, 1, num_register_tokens, dec_embed_dim))
        nn.init.normal_(self.register_token_sat, std=1e-6)

        # Satellite position encoding
        self.sat_pos_embedder = FourierEmbedder(
            in_dim=2, embed_dim=dec_embed_dim, num_freqs=64, scale=1.5)

        # Post-decoder satellite injection
        self.use_sat_inject = use_sat_inject
        self.use_sat_cross_attn = use_sat_cross_attn
        self.use_bidirectional_cross_attn = use_bidirectional_cross_attn
        if use_sat_inject:
            self.sat_inject = SatPatchInjection(dim=1024, num_register=num_register_tokens)
        if use_sat_cross_attn:
            self.sat_cross_attn = SpatialSatCrossAttention(
                dim=1024, hidden_dim=512, num_heads=8)
        if use_bidirectional_cross_attn:
            self.bidirectional_cross_attn = BidirectionalSatCrossAttention(
                dim=1024, hidden_dim=512, num_heads=8)

        # Decoder-level adapters
        self.use_decoder_adapters = use_decoder_adapters
        if use_decoder_adapters:
            self.adapter_layers = adapter_layers
            self.decoder_adapters = nn.ModuleList([
                DecoderSatAdapter(dim=dec_embed_dim, num_heads=16)
                for _ in adapter_layers])

        # Exp20: parallel trainable sat decoder (mirrors frozen decoder architecture)
        self.use_sat_parallel_decoder = use_sat_parallel_decoder
        if use_sat_parallel_decoder:
            self.sat_decoder = nn.ModuleList([
                BlockRope(
                    dim=dec_embed_dim, num_heads=dec_num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=True, proj_bias=True, ffn_bias=True, drop_path=0.0,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    act_layer=nn.GELU, ffn_layer=Mlp, init_values=0.01,
                    qk_norm=True, attn_class=FlashAttentionRope, rope=self.rope
                ) for _ in range(dec_depth)])

        # Exp21: lightweight adapter on camera_hidden for sat views
        self.use_sat_camera_adapter = use_sat_camera_adapter
        if use_sat_camera_adapter:
            self.sat_camera_adapter = SatCameraAdapter(dim=512)

        # Exp22: parallel trainable camera decoder for sat views
        self.use_sat_parallel_camera = use_sat_parallel_camera
        if use_sat_parallel_camera:
            self.sat_camera_decoder = TransformerDecoder(
                in_dim=2 * dec_embed_dim, dec_embed_dim=1024,
                dec_num_heads=16, out_dim=512, rope=self.rope, use_checkpoint=False)

        # Exp23: full parallel sat camera branch (decoder + head, fully trainable, direct prediction)
        self.use_sat_camera_branch = use_sat_camera_branch
        if use_sat_camera_branch:
            self.sat_camera_decoder_full = TransformerDecoder(
                in_dim=2 * dec_embed_dim, dec_embed_dim=1024,
                dec_num_heads=16, out_dim=512, rope=self.rope, use_checkpoint=False)
            self.sat_camera_head = CameraHead(dim=512)

        # Exp24: full parallel sat point branch (decoder + head, fully trainable, direct prediction)
        self.use_sat_point_branch = use_sat_point_branch
        if use_sat_point_branch:
            self.sat_point_decoder = TransformerDecoder(
                in_dim=2 * dec_embed_dim, dec_embed_dim=1024,
                dec_num_heads=16, out_dim=1024, rope=self.rope)
            self.sat_point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # ===================== Load Weights & Freeze =====================

        if ckpt is not None:
            state = torch.load(ckpt, weights_only=False, map_location='cpu')
            res = self.load_state_dict(state, strict=False)
            print(f'[Pi3FrozenSat] Load checkpoint from {ckpt}: {res}')
            del state
            torch.cuda.empty_cache()
        elif load_pi3:
            self._load_pi3_and_freeze()

        if freeze_encoder:
            freeze_all_params([self.encoder])

    def _load_pi3_and_freeze(self):
        """Load original Pi3 weights, initialize sat tokens, freeze backbone."""
        pi3_weight = load_file('ckpts/Pi3/model_pi3.safetensors')

        # Original Pi3 has register_token (1,1,5,dim) → load into register_token_grd
        if 'register_token' in pi3_weight:
            reg = pi3_weight.pop('register_token')  # (1,1,5,dim)
            pi3_weight['register_token_grd'] = reg
            # Initialize sat register token from same weights
            self.register_token_sat.data.copy_(reg)

        res = self.load_state_dict(pi3_weight, strict=False)
        print(f"[Pi3FrozenSat] Loading Pi3 weights: {res}")
        if res.missing_keys:
            print(f"[Pi3FrozenSat] Missing keys (new sat modules): {res.missing_keys}")

        # Exp20: warm-start sat_decoder from frozen decoder weights
        if self.use_sat_parallel_decoder:
            for sat_blk, frozen_blk in zip(self.sat_decoder, self.decoder):
                sat_blk.load_state_dict(frozen_blk.state_dict())
            print(f"[Pi3FrozenSat] sat_decoder warm-started from frozen decoder ({len(self.sat_decoder)} blocks).")

        # Exp22: warm-start sat_camera_decoder from frozen camera_decoder
        if self.use_sat_parallel_camera:
            self.sat_camera_decoder.load_state_dict(self.camera_decoder.state_dict())
            print("[Pi3FrozenSat] sat_camera_decoder warm-started from frozen camera_decoder.")

        # Exp23: warm-start sat_camera_decoder_full + sat_camera_head from frozen counterparts
        if self.use_sat_camera_branch:
            self.sat_camera_decoder_full.load_state_dict(self.camera_decoder.state_dict())
            self.sat_camera_head.load_state_dict(self.camera_head.state_dict())
            print("[Pi3FrozenSat] sat_camera_decoder_full + sat_camera_head warm-started from frozen counterparts.")

        # Exp24: warm-start sat_point_decoder + sat_point_head from frozen counterparts
        if self.use_sat_point_branch:
            self.sat_point_decoder.load_state_dict(self.point_decoder.state_dict())
            self.sat_point_head.load_state_dict(self.point_head.state_dict())
            print("[Pi3FrozenSat] sat_point_decoder + sat_point_head warm-started from frozen counterparts.")

        # Freeze all original Pi3 components
        freeze_all_params([
            self.decoder,
            self.point_decoder, self.point_head,
            self.camera_decoder, self.camera_head,
            self.register_token_grd,
        ])
        if self.train_conf and self.conf_decoder is not None:
            freeze_all_params([self.conf_decoder, self.conf_head])
        print("[Pi3FrozenSat] Backbone frozen. Trainable: sat adapters only.")

    def decode(self, hidden, N, H, W, is_sat_mask=None, sat_pos_embed=None):
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []
        hidden = hidden.reshape(B * N, hw, -1)

        # Assign register tokens: sat views → trainable sat tokens, others → frozen grd tokens
        if is_sat_mask is not None:
            reg_sat = self.register_token_sat.expand(B, N, -1, -1)
            reg_grd = self.register_token_grd.expand(B, N, -1, -1)
            mask = is_sat_mask.to(hidden.device).view(B, N, 1, 1)
            register_token = torch.where(mask, reg_sat, reg_grd)
            register_token = register_token.reshape(B * N, *self.register_token_grd.shape[-2:])
        else:
            register_token = self.register_token_grd.repeat(B, N, 1, 1).reshape(
                B * N, *self.register_token_grd.shape[-2:])

        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        # Add satellite position encoding
        if sat_pos_embed is not None and is_sat_mask is not None:
            if self.patch_start_idx > 0:
                pad = torch.zeros(B, self.patch_start_idx, sat_pos_embed.shape[-1],
                                  device=sat_pos_embed.device, dtype=sat_pos_embed.dtype)
                sat_pos_embed = torch.cat([pad, sat_pos_embed], dim=1)
            hidden = hidden.reshape(B, N, hw, -1)
            sat_mask = is_sat_mask.to(hidden.device).unsqueeze(-1).unsqueeze(-1).float()
            hidden = hidden + sat_mask * sat_pos_embed[:, None, :, :]
            hidden = hidden.reshape(B * N, hw, -1)

        # Positional encoding
        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H // self.patch_size, W // self.patch_size, hidden.device)
        if self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2,
                                      device=hidden.device, dtype=pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # Build adapter lookup
        adapter_map = {}
        if self.use_decoder_adapters:
            for idx, layer_i in enumerate(self.adapter_layers):
                adapter_map[layer_i] = idx

        # Exp20: pre-compute split indices for parallel sat decoder
        use_parallel = self.use_sat_parallel_decoder and is_sat_mask is not None
        if use_parallel:
            # Flat B*N indexing for intra-view (robust to per-batch-element variation)
            is_sat_flat = is_sat_mask.reshape(-1)
            grd_bn_idx = torch.where(~is_sat_flat)[0]
            sat_bn_idx = torch.where(is_sat_flat)[0]
            # Cross-view uses per-view N indexing (assumes uniform mask across batch)
            is_sat_one = is_sat_mask[0]
            grd_n_idx = torch.where(~is_sat_one)[0]
            sat_n_idx = torch.where(is_sat_one)[0]
            n_grd = grd_n_idx.numel()
            n_sat = sat_n_idx.numel()

        # Decoder loop
        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B * N, hw, -1)
                hidden = hidden.reshape(B * N, hw, -1)
            else:
                pos = pos.reshape(B, N * hw, -1)
                hidden = hidden.reshape(B, N * hw, -1)

            if use_parallel:
                sat_blk = self.sat_decoder[i]
                if i % 2 == 0:
                    # Intra-view self-attn — split by view type; guard empty paths
                    new_hidden = hidden.clone()
                    if grd_bn_idx.numel() > 0:
                        grd_h = hidden.index_select(0, grd_bn_idx)
                        grd_p = pos.index_select(0, grd_bn_idx)
                        grd_h = self._apply_block_ckpt(blk, grd_h, grd_p, i)
                        new_hidden = new_hidden.index_copy(0, grd_bn_idx, grd_h)
                    if sat_bn_idx.numel() > 0:
                        sat_h = hidden.index_select(0, sat_bn_idx)
                        sat_p = pos.index_select(0, sat_bn_idx)
                        sat_h = self._apply_block_ckpt(sat_blk, sat_h, sat_p, i)
                        new_hidden = new_hidden.index_copy(0, sat_bn_idx, sat_h)
                    hidden = new_hidden
                else:
                    # Cross-view attn — ground-only path (frozen) + sat-aware path (trainable, all views)
                    hidden_per_view = hidden.reshape(B, N, hw, -1)
                    pos_per_view = pos.reshape(B, N, hw, -1)
                    new_per_view = hidden_per_view.clone()
                    if n_grd > 0:
                        grd_only_h = hidden_per_view.index_select(1, grd_n_idx).reshape(B, n_grd * hw, -1)
                        grd_only_p = pos_per_view.index_select(1, grd_n_idx).reshape(B, n_grd * hw, -1)
                        grd_only_out = self._apply_block_ckpt(blk, grd_only_h, grd_only_p, i)
                        grd_only_out = grd_only_out.reshape(B, n_grd, hw, -1)
                        new_per_view = new_per_view.index_copy(1, grd_n_idx, grd_only_out)
                    if n_sat > 0:
                        full_out = self._apply_block_ckpt(sat_blk, hidden, pos, i)
                        full_out_per_view = full_out.reshape(B, N, hw, -1)
                        sat_out = full_out_per_view.index_select(1, sat_n_idx)
                        new_per_view = new_per_view.index_copy(1, sat_n_idx, sat_out)
                    hidden = new_per_view.reshape(B, N * hw, -1)
            else:
                hidden = self._apply_block_ckpt(blk, hidden, pos, i)

            # Apply trainable adapter after this layer (Exp18)
            if i in adapter_map and is_sat_mask is not None:
                hidden = hidden.reshape(B * N, hw, -1)
                hidden = self.decoder_adapters[adapter_map[i]](hidden, is_sat_mask)

            if i + 1 in [len(self.decoder) - 1, len(self.decoder)]:
                final_output.append(hidden.reshape(B * N, hw, -1))

        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B * N, hw, -1)

    def _apply_block_ckpt(self, blk, hidden, pos, layer_i):
        """Apply BlockRope with optional gradient checkpointing."""
        if layer_i >= self.num_dec_blk_not_to_checkpoint and self.training:
            return checkpoint(blk, hidden, xpos=pos, use_reentrant=False)
        return blk(hidden, xpos=pos)

    def forward(self, imgs, is_sat_mask=None):
        imgs = (imgs - self.image_mean) / self.image_std
        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14

        # Encode (frozen)
        imgs = imgs.reshape(B * N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)
        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        # Satellite position encoding (trainable)
        has_sat = is_sat_mask is not None and is_sat_mask.any()
        sat_pos_embed = None
        if has_sat:
            y_steps = torch.linspace(-1, 1, patch_h, device=hidden.device, dtype=hidden.dtype)
            x_steps = torch.linspace(-1, 1, patch_w, device=hidden.device, dtype=hidden.dtype)
            grid_y, grid_x = torch.meshgrid(y_steps, x_steps, indexing='ij')
            pos_input = torch.stack([grid_x, grid_y], dim=-1)
            sat_pos_embed = self.sat_pos_embedder(pos_input.unsqueeze(0).expand(B, -1, -1, -1))
            sat_pos_embed = sat_pos_embed.reshape(B, patch_h * patch_w, -1)

        # Decode (frozen decoder + optional trainable adapters)
        hidden, pos = self.decode(hidden, N, H, W, is_sat_mask, sat_pos_embed)

        # Point prediction
        local_points_precomputed = None  # Exp24 path: skip the autocast point_head call
        point_hidden = None
        if has_sat and self.use_sat_point_branch:
            # Exp24: fully split point decoder AND head — sat path is independent end-to-end
            BN_pt = hidden.shape[0]
            is_sat_flat = is_sat_mask.reshape(-1)
            grd_bn_pt = torch.where(~is_sat_flat)[0]
            sat_bn_pt = torch.where(is_sat_flat)[0]
            local_points_precomputed = torch.zeros(
                BN_pt, H, W, 3, device=hidden.device, dtype=torch.float32)
            if grd_bn_pt.numel() > 0:
                grd_h = hidden.index_select(0, grd_bn_pt)
                grd_p = pos.index_select(0, grd_bn_pt)
                grd_pt_hidden = self.point_decoder(grd_h, xpos=grd_p)
                with torch.amp.autocast(device_type='cuda', enabled=False):
                    grd_ret = self.point_head(
                        [grd_pt_hidden[:, self.patch_start_idx:].float()], (H, W))
                    grd_ret = grd_ret.reshape(grd_pt_hidden.shape[0], H, W, -1)
                    grd_xy, grd_z = grd_ret.split([2, 1], dim=-1)
                    grd_z = torch.exp(grd_z)
                    grd_local = torch.cat([grd_xy * grd_z, grd_z], dim=-1)
                local_points_precomputed = local_points_precomputed.index_copy(
                    0, grd_bn_pt, grd_local)
            if sat_bn_pt.numel() > 0:
                sat_h = hidden.index_select(0, sat_bn_pt)
                sat_p = pos.index_select(0, sat_bn_pt)
                sat_pt_hidden = self.sat_point_decoder(sat_h, xpos=sat_p)
                with torch.amp.autocast(device_type='cuda', enabled=False):
                    sat_ret = self.sat_point_head(
                        [sat_pt_hidden[:, self.patch_start_idx:].float()], (H, W))
                    sat_ret = sat_ret.reshape(sat_pt_hidden.shape[0], H, W, -1)
                    sat_xy, sat_z = sat_ret.split([2, 1], dim=-1)
                    sat_z = torch.exp(sat_z)
                    sat_local = torch.cat([sat_xy * sat_z, sat_z], dim=-1)
                local_points_precomputed = local_points_precomputed.index_copy(
                    0, sat_bn_pt, sat_local)
        else:
            point_hidden = self.point_decoder(hidden, xpos=pos)
            if has_sat:
                if self.use_sat_inject:
                    point_hidden = self.sat_inject(point_hidden, is_sat_mask)
                if self.use_sat_cross_attn:
                    point_hidden = self.sat_cross_attn(point_hidden, is_sat_mask)
                if self.use_bidirectional_cross_attn:
                    point_hidden = self.bidirectional_cross_attn(point_hidden, is_sat_mask)

        if self.train_conf:
            conf_hidden = self.conf_decoder(hidden, xpos=pos)

        # Camera prediction
        camera_poses_precomputed = None  # Exp23 path: skip the autocast camera_head call
        camera_hidden = None
        if has_sat and self.use_sat_camera_branch:
            # Exp23: fully split decoder AND head — sat path is independent end-to-end (no gate, direct predict)
            BN_cam = hidden.shape[0]
            is_sat_flat = is_sat_mask.reshape(-1)
            grd_bn_cam = torch.where(~is_sat_flat)[0]
            sat_bn_cam = torch.where(is_sat_flat)[0]
            camera_poses_precomputed = torch.zeros(
                BN_cam, 4, 4, device=hidden.device, dtype=torch.float32)
            if grd_bn_cam.numel() > 0:
                grd_h = hidden.index_select(0, grd_bn_cam)
                grd_p = pos.index_select(0, grd_bn_cam)
                grd_cam_hidden = self.camera_decoder(grd_h, xpos=grd_p)
                with torch.amp.autocast(device_type='cuda', enabled=False):
                    grd_poses = self.camera_head(
                        grd_cam_hidden[:, self.patch_start_idx:].float(), patch_h, patch_w)
                camera_poses_precomputed = camera_poses_precomputed.index_copy(
                    0, grd_bn_cam, grd_poses)
            if sat_bn_cam.numel() > 0:
                sat_h = hidden.index_select(0, sat_bn_cam)
                sat_p = pos.index_select(0, sat_bn_cam)
                sat_cam_hidden = self.sat_camera_decoder_full(sat_h, xpos=sat_p)
                with torch.amp.autocast(device_type='cuda', enabled=False):
                    sat_poses = self.sat_camera_head(
                        sat_cam_hidden[:, self.patch_start_idx:].float(), patch_h, patch_w)
                camera_poses_precomputed = camera_poses_precomputed.index_copy(
                    0, sat_bn_cam, sat_poses)
        elif has_sat and self.use_sat_parallel_camera:
            # Exp22: split decoder, shared frozen head
            BN_cam = hidden.shape[0]
            is_sat_flat = is_sat_mask.reshape(-1)
            grd_bn_cam = torch.where(~is_sat_flat)[0]
            sat_bn_cam = torch.where(is_sat_flat)[0]
            camera_hidden = torch.zeros(
                BN_cam, hidden.shape[1], 512,
                device=hidden.device, dtype=hidden.dtype)
            if grd_bn_cam.numel() > 0:
                grd_h = hidden.index_select(0, grd_bn_cam)
                grd_p = pos.index_select(0, grd_bn_cam)
                grd_cam = self.camera_decoder(grd_h, xpos=grd_p)
                camera_hidden = camera_hidden.index_copy(0, grd_bn_cam, grd_cam)
            if sat_bn_cam.numel() > 0:
                sat_h = hidden.index_select(0, sat_bn_cam)
                sat_p = pos.index_select(0, sat_bn_cam)
                sat_cam = self.sat_camera_decoder(sat_h, xpos=sat_p)
                camera_hidden = camera_hidden.index_copy(0, sat_bn_cam, sat_cam)
        else:
            camera_hidden = self.camera_decoder(hidden, xpos=pos)

        # Exp21: adapter on camera_hidden for sat views only (skip if Exp23)
        if camera_hidden is not None and has_sat and self.use_sat_camera_adapter:
            camera_hidden = self.sat_camera_adapter(camera_hidden, is_sat_mask)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            if local_points_precomputed is not None:
                # Exp24: local_points already computed via split point_decoder/head paths
                local_points = local_points_precomputed.reshape(B, N, H, W, -1)
            else:
                point_hidden = point_hidden.float()
                ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
                xy, z = ret.split([2, 1], dim=-1)
                z = torch.exp(z)
                local_points = torch.cat([xy * z, z], dim=-1)

            if self.train_conf:
                conf_hidden = conf_hidden.float()
                conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            else:
                conf = None

            if camera_poses_precomputed is not None:
                # Exp23: poses already computed via split decoder/head paths
                camera_poses = camera_poses_precomputed.reshape(B, N, 4, 4)
            else:
                camera_hidden = camera_hidden.float()
                camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)

            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

        return dict(
            points=points,
            sat_ori_xy=None,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
            global_points=None,
            sat_mpp=None
        )
