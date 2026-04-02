from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d, ContextTransformerDecoder
from .layers.camera_head import CameraHead
from .layers.infinidepth_conv import ImplicitHead, BasicEncoder
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg
from .sat_position import FourierEmbedder
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file
def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            # module is directly a parameter
            module.requires_grad = False

class Pi3(nn.Module):
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
            load_pi3=True,
            freeze_encoder=True,
            use_global_points=False,
            train_conf=False,
            num_dec_blk_not_to_checkpoint=4,
            ckpt=None,
            default_query_count=112 * 112,
            basic_encoder_dim: int = 128,
        ):
        super().__init__()

        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope=None
        if self.pos_type.startswith('rope'): # eg rope100 
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError
        
        self.sat_pos_embedder = FourierEmbedder(
            in_dim=2, 
            embed_dim=self.encoder.embed_dim, 
            num_freqs=64, 
            scale=1.5 
        )

        # ----------------------
        #        Decoder
        # ----------------------
        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim


        # ----------------------
        #     Query：patch token 序列 + (u,v) → 局部位姿系 3D 点（ImplicitHead 内双线性采样 + MLP）
        # ----------------------
        self.basic_encoder_dim = int(basic_encoder_dim)
        self.basic_encoder = BasicEncoder(input_dim=3, output_dim=self.basic_encoder_dim, stride=4)
        self.query_implicit_head = ImplicitHead(
            hidden_dim=dec_embed_dim,
            basic_dim=self.basic_encoder_dim,
            fusion_type="gated",
            out_dim=3,
            hidden_list=[max(dec_embed_dim, 512), 256, 32],
        )
        self.default_query_count = int(default_query_count)

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Local Points Decoder：先用 point_decoder 细化 token，再在 query 处采样
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False
        )
        self.camera_head = CameraHead(dim=512)

        # ----------------------
        #  Global Points Decoder
        # ----------------------
        self.use_global_points = use_global_points
        if use_global_points:
            self.global_points_decoder = ContextTransformerDecoder(
                in_dim=2*self.dec_embed_dim, 
                dec_embed_dim=1024,
                dec_num_heads=16,
                out_dim=1024,
                rope=self.rope,
            )
            self.global_point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        self.train_conf = train_conf
        if self.train_conf:
            self.conf_head = nn.Sequential(
                nn.Linear(self.dec_embed_dim, 512),
                nn.GELU(),
                nn.Linear(512, 1)
            )
            freeze_all_params([
                self.encoder, self.decoder, self.point_decoder,
                self.query_implicit_head,
                self.basic_encoder,
                self.camera_decoder, self.camera_head, self.register_token
            ])
        else:
            self.conf_head = None

        if freeze_encoder:
            print('Freezing the encoder.')
            freeze_all_params([self.encoder])

        self.num_dec_blk_not_to_checkpoint = num_dec_blk_not_to_checkpoint

        if ckpt is not None:
            checkpoint = torch.load(ckpt, weights_only=False, map_location='cpu')

            res = self.load_state_dict(checkpoint, strict=False)
            print(f'[Pi3] Load checkpoints from {ckpt}: {res}')

            del checkpoint
            torch.cuda.empty_cache()
        elif load_pi3:
            pi3_weight = load_file('ckpts/Pi3/model_pi3.safetensors')

            # =======================================================
            # 1. [新增] 解决 register_token 形状不匹配问题
            #    将 (1, 1, tokens, dim) -> 复制为 (1, 2, tokens, dim)
            # =======================================================
            if 'register_token' in pi3_weight:
                reg_token = pi3_weight['register_token']
                # 检查维度：如果权重里的第1维是1，但当前模型需要2
                if reg_token.shape[1] == 1 and self.register_token.shape[1] == 2:
                    print(f"[Pi3] Expanding register_token from {reg_token.shape} to {self.register_token.shape}")
                    # 在第1维复制一份：view0 和 view1 初始权重相同
                    pi3_weight['register_token'] = reg_token.repeat(1, 2, 1, 1)

            res = self.load_state_dict(pi3_weight, strict=False)
            print("Loading pi3 weights", res)
            if res.unexpected_keys:
                print("[Pi3] Unexpected keys (in ckpt but not in model):", res.unexpected_keys)
            if res.missing_keys:
                print("================================================")
                print("[Pi3] Missing keys (in model but not in ckpt):", res.missing_keys)

    def decode(self, hidden, N, H, W, is_sat_mask, sat_pos_embed=None,):
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []
        
        hidden = hidden.reshape(B*N, hw, -1)

        # 使用 is_sat_mask 精确区分：所有卫星视图共用同一组 register_token，其它视图共用另一组
        # 1. 取出两个模板 token 组：sat / non-sat -> shape (1, 1, tokens, dim)
        reg_token_sat = self.register_token[:, 0:1]
        reg_token_non_sat = self.register_token[:, 1:2]

        # 2. 扩展到 (B, N, tokens, dim)，按 is_sat_mask 选择
        reg_token_sat = reg_token_sat.expand(B, N, -1, -1)
        reg_token_non_sat = reg_token_non_sat.expand(B, N, -1, -1)

        mask = is_sat_mask.to(hidden.device).view(B, N, 1, 1)
        register_token = torch.where(mask, reg_token_sat, reg_token_non_sat)
        # 展平为 (B*N, tokens, dim) 以适配后续的计算
        register_token = register_token.reshape(B*N, *self.register_token.shape[-2:])

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        # ======================================================================
        # 1. 补齐 register tokens 的长度
        # 2. 【核心修改】在进入循环前，一次性注入卫星位置编码
        # ======================================================================
        if sat_pos_embed is not None:
            if self.patch_start_idx > 0:
                pad_zeros = torch.zeros(
                    B, self.patch_start_idx, sat_pos_embed.shape[-1],
                    device=sat_pos_embed.device, dtype=sat_pos_embed.dtype
                )
                sat_pos_embed = torch.cat([pad_zeros, sat_pos_embed], dim=1)  # (B, hw, dim)

            # 暂时变形为 (B, N, hw, dim)，根据 is_sat_mask 精准定位卫星视图
            hidden = hidden.reshape(B, N, hw, -1)

            # is_sat_mask: (B, N) bool → (B, N, 1, 1) float
            sat_mask = is_sat_mask.to(hidden.device).unsqueeze(-1).unsqueeze(-1).float()
            sat_pos = sat_pos_embed[:, None, :, :]  # (B, 1, hw, dim)
            hidden = hidden + sat_mask * sat_pos

            # 重新展平为 (B*N, hw, dim)，准备进入 Decoder 循环
            hidden = hidden.reshape(B*N, hw, -1)
        # ======================================================================

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
       
        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B*N, hw, -1)
                hidden = hidden.reshape(B*N, hw, -1)
            else:
                pos = pos.reshape(B, N*hw, -1)
                hidden = hidden.reshape(B, N*hw, -1)

            if i >= self.num_dec_blk_not_to_checkpoint and self.training:
                hidden = checkpoint(blk, hidden, xpos=pos, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=pos)

            if i+1 in [len(self.decoder)-1, len(self.decoder)]:
                final_output.append(hidden.reshape(B*N, hw, -1))

        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1)
    
    def forward(self, 
                imgs, 
                is_sat_mask, 
                queries=None,
                dense: bool = False, 
                isTrain: bool = True
        ):  # [关键修改] 加入 queries 和 is_sat_mask 参数
        imgs_01 = imgs
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape

        patch_h, patch_w = H // 14, W // 14
        
        # encode by dinov2
        frames = imgs.reshape(B*N, _, H, W)
        dinov2_hidden = self.encoder(frames, is_training=True)

        if isinstance(dinov2_hidden, dict):
            dinov2_hidden = dinov2_hidden["x_norm_patchtokens"]

        # ==========================================================
        # 1. 只生成卫星位置编码，不在这里直接与 hidden 叠加
        # ==========================================================
        y_steps = (torch.arange(patch_h, device=dinov2_hidden.device, dtype=dinov2_hidden.dtype) + 0.5) / patch_h * 2.0 - 1.0
        x_steps = (torch.arange(patch_w, device=dinov2_hidden.device, dtype=dinov2_hidden.dtype) + 0.5) / patch_w * 2.0 - 1.0
        grid_y, grid_x = torch.meshgrid(y_steps, x_steps, indexing='ij')
        
        pos_input = torch.stack([grid_x, grid_y], dim=-1) # (H, W, 2)
        sat_pos_embed = self.sat_pos_embedder(pos_input.unsqueeze(0).expand(B, -1, -1, -1))
        
        # 保存下来供 decode 循环使用
        sat_pos_embed = sat_pos_embed.reshape(B, patch_h * patch_w, -1)
        # 将 sat_pos_embed 传给 decode，并用 is_sat_mask 控制哪些视图叠加卫星位置编码
        hidden, pos = self.decode(dinov2_hidden, N, H, W, sat_pos_embed=sat_pos_embed, is_sat_mask=is_sat_mask)

        # ==========================================================
        # 2. 生成/处理 Queries (u, v)
        # - 若外部提供 queries（来自 dataset），则直接使用
        # - 否则保留原有稀疏/稠密采样逻辑
        # ==========================================================
        if queries is not None and dense is False:
            queries = queries.reshape(B * N, -1, 2)
            query_per_view = queries.shape[1]
        else:
            # 使用像素中心约定 (pixel-center convention)，与 dataset 采样一致：
            #   u = (pixel_x + 0.5) / W,  v = (pixel_y + 0.5) / H
            x_steps = (torch.arange(W, device=hidden.device, dtype=hidden.dtype) + 0.5) / W
            y_steps = (torch.arange(H, device=hidden.device, dtype=hidden.dtype) + 0.5) / H
            grid_y, grid_x = torch.meshgrid(y_steps, x_steps, indexing='ij')
            dense_queries = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)  # (H*W, 2)

            if dense:
                query_per_view = dense_queries.shape[0]
                queries = dense_queries.unsqueeze(0).expand(B * N, -1, -1).contiguous() # (B*N, H*W, 2)
            else:
                query_per_view = min(self.default_query_count, dense_queries.shape[0])
                rand_matrix = torch.rand(B * N, dense_queries.shape[0], device=hidden.device)
                idx = rand_matrix.argsort(dim=1)[:, :query_per_view]
                queries = dense_queries[idx] # (B*N, query_per_view, 2)

        # ==========================================================
        # 3. point_decoder(hidden) patch token + queries：ImplicitHead 在 query 处 grid_sample，MLP → (x,y,z)
        # ==========================================================
        point_tokens = self.point_decoder(hidden, xpos=pos)
        point_feat = point_tokens[:, self.patch_start_idx:]

        grid = (queries * 2.0 - 1.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6).unsqueeze(1)

        def sample_feat(feat_map):
            sampled = F.grid_sample(feat_map, grid, mode='bilinear', align_corners=False)
            return sampled.squeeze(2).transpose(1, 2)

        point_map_e = point_feat.transpose(1, 2).reshape(B * N, self.dec_embed_dim, patch_h, patch_w)
        q_for_conf = sample_feat(point_map_e) if self.train_conf else None

        x_basic = imgs_01.reshape(B * N, _, H, W)
        basic_feat_map = self.basic_encoder(x_basic)  # (B*N, basic_dim, H/4, W/4)

        local_points = self.query_implicit_head(point_feat, basic_feat_map, patch_h, patch_w, queries)
        local_points = local_points.reshape(B, N, query_per_view, 3)
        
        # 3.6. 处理相机hidden
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            local_points = local_points.float()

            if self.train_conf:
                conf = self.conf_head(q_for_conf.float()).reshape(B, N, query_per_view, -1)
            else:
                conf = None
                
            # camera
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)

            # unproject local points using camera poses
            points = torch.einsum('bnij, bnqj -> bnqi', camera_poses, homogenize_points(local_points))[..., :3]

        if dense and query_per_view == H * W and isTrain is False:
            points_out = points.reshape(B, N, H, W, 3)
            local_points_out = local_points.reshape(B, N, H, W, 3)
            query_uv_out = queries.reshape(B, N, H, W, 2)
            conf_out = conf.reshape(B, N, H, W, -1) if conf is not None else None
        else:
            points_out = points
            local_points_out = local_points
            query_uv_out = queries.reshape(B, N, query_per_view, 2)
            conf_out = conf

        return dict[str, Any | None](
            points=points_out,
            local_points=local_points_out,
            query_uv=query_uv_out,
            conf=conf_out,
            camera_poses=camera_poses,
            global_points=None
        )
