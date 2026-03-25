from typing import Any
import torch
import torch.nn as nn
from functools import partial
from einops import rearrange

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d, ContextTransformerDecoder
from .layers.camera_head import CameraHead
from .layers.conv_head import ConvHead, InfiniDepthFusion
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg
from .sat_position import FourierEmbedder
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file
from .query import DecoderBlock, PatchEmbeddingFast, ContinuousRoPE2D, MultiScaleQueryEmbedder

MIN_MPP = 0.01
MAX_MPP = 0.1

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
            query_decoder_depth=4,
            ckpt=None,
            default_query_count=112 * 112,
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
        #     Query Decoder
        # ----------------------
        self.query_hidden_project = nn.Linear(dec_embed_dim*2, dec_embed_dim)
        self.query_pos_embedder = FourierEmbedder(
            in_dim=2, 
            embed_dim=dec_embed_dim, 
            num_freqs=64, 
            scale=30.0, 
            include_input=True,
            zero_init=True
        )
        self.conv_head = ConvHead(
            dim_in=self.dec_embed_dim,      # 输入维度 (1024 * 2，因为后面会和 hidden 拼接)
            dim_out=[128],                  # 最终输出的 f_high 维度
            dim_proj=512,                   # f_low 的维度 (第一层投影)
            dim_upsample=[256, 128, 128],   # 上采样阶段的通道数 (f_mid 会截取第一个 256)
            last_conv_channels=128,         # ⚠️ 必须修改！防止 128 维输出被 32 维瓶颈卡死
            dim_times_res_block_hidden=2,
            num_res_blocks=2,
            res_block_norm='group_norm',
            projects=nn.Linear(self.dec_embed_dim * 2, 512), # 将 DINOv2 的 1024 * 2 维压缩成 f_low
            using_uv=True                   # ⚠️ 注入极其关键的 UV 坐标空间先验！
        )
        self.ms_fusion = InfiniDepthFusion(dims=[128, 256, 512], target_dim=dec_embed_dim)
        self.query_rope = ContinuousRoPE2D(freq=100.0)
        self.query_decoder = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio, True, 0.0, 0.0, rope=self.query_rope, use_self_attn=False)
            for _ in range(query_decoder_depth)
        ])
        self.query_norm = nn.LayerNorm(dec_embed_dim)
        # self.patch_embed = PatchEmbeddingFast(patch_size=9, embed_dim=dec_embed_dim)
        self.query_token_sat = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))
        self.query_token_grd = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))
        self.default_query_count = int(default_query_count)

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.grd_point_head = nn.Sequential(
            nn.Linear(dec_embed_dim, 512),
            nn.GELU(),
            nn.Linear(512, 3)
        )
        self.sat_point_head = nn.Sequential(
            nn.Linear(dec_embed_dim, 512),
            nn.GELU(),
            nn.Linear(512, 2)  # [修改] 从 3 改为 2：输出 (log_mpp, z)
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
                self.encoder, self.decoder, self.query_pos_embedder, self.query_decoder,
                self.query_norm, self.conv_head, self.ms_fusion, self.query_token_sat, self.query_token_grd, # <-- 更新这里
                self.grd_point_head, self.sat_point_head,
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

    def decode(self, hidden, N, H, W, sat_pos_embed=None, is_sat_mask: torch.Tensor | None = None):
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []
        
        hidden = hidden.reshape(B*N, hw, -1)

        # Original code:
        # register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])

        # 1. 取出专门给 View 0 (卫星图) 的 Token -> shape (1, 1, tokens, dim)
        reg_token_view0 = self.register_token[:, 0:1]

        # 2. 取出给 View 1~N (地面图) 的 Token -> shape (1, 1, tokens, dim)
        reg_token_others = self.register_token[:, 1:2]

        # 3. 扩展 Batch 维度
        # View 0: 每个 Batch 有 1 张 -> (B, 1, tokens, dim)
        reg_token_view0 = reg_token_view0.expand(B, 1, -1, -1)

        # 4. Others: 每个 Batch 有 N-1 张 -> (B, N-1, tokens, dim)
        if N > 1:
            reg_token_others = reg_token_others.expand(B, N - 1, -1, -1)
            # 4. 在 N 这个维度拼接 -> (B, N, tokens, dim)
            register_token = torch.cat([reg_token_view0, reg_token_others], dim=1)
        else:
            # 如果只有一个视图，就只用 View 0 的 token
            register_token = reg_token_view0

        # 5. 展平为 (B*N, tokens, dim) 以适配后续的计算
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

            if is_sat_mask is not None:
                # is_sat_mask: (B, N) bool → (B, N, 1, 1) float
                sat_mask = is_sat_mask.to(hidden.device).unsqueeze(-1).unsqueeze(-1).float()
                sat_pos = sat_pos_embed[:, None, :, :]  # (B, 1, hw, dim)
                hidden = hidden + sat_mask * sat_pos
            else:
                # 兼容旧逻辑：没有 mask 时，默认第 0 个视图为卫星图
                hidden[:, 0] = hidden[:, 0] + sat_pos_embed

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
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape

        patch_h, patch_w = H // 14, W // 14
        
        # encode by dinov2
        frames = imgs.reshape(B*N, _, H, W)
        hidden = self.encoder(frames, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        # ==========================================================
        # 1. 只生成卫星位置编码，不在这里直接与 hidden 叠加
        # ==========================================================
        y_steps = torch.linspace(-1, 1, patch_h, device=hidden.device, dtype=hidden.dtype)
        x_steps = torch.linspace(-1, 1, patch_w, device=hidden.device, dtype=hidden.dtype)
        grid_y, grid_x = torch.meshgrid(y_steps, x_steps, indexing='ij')
        
        pos_input = torch.stack([grid_x, grid_y], dim=-1) # (H, W, 2)
        sat_pos_embed = self.sat_pos_embedder(pos_input.unsqueeze(0).expand(B, -1, -1, -1))
        
        # 保存下来供 decode 循环使用
        sat_pos_embed = sat_pos_embed.reshape(B, patch_h * patch_w, -1)
        # 将 sat_pos_embed 传给 decode，并用 is_sat_mask 控制哪些视图叠加卫星位置编码
        hidden, pos = self.decode(hidden, N, H, W, sat_pos_embed, is_sat_mask=is_sat_mask)

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

        num_queries = queries.shape[1]

        # ==========================================================
        # 3. D4RT 核心：交叉注意力查询 (Cross-Attention Decoding)
        # ==========================================================
        # 3.1 傅里叶编码 Query 坐标
        query_pos_embeddings = self.query_pos_embedder(queries * 2.0 - 1.0) # (B*N, Num_queries, dim)

        # 3.2 Local RGB patch embedding
        # patch_rgb_embeddings = self.patch_embed(imgs, queries)  # (B*N, Num_queries, embed_dim)
        
        # 3.2 InfiniDepth 风格的多尺度局部特征采样
        # 3.2.1 获取多尺度金字塔特征
        feat_low, feat_mid, feat_high = self.conv_head(hidden[:, self.patch_start_idx:], image=frames)
        
        # 3.2.2 连续坐标网格采样
        grid = queries * 2.0 - 1.0  
        grid = grid.unsqueeze(1)    # [B*N, 1, Q, 2]
        
        # 写一个小函数批量采样，让代码保持极度整洁
        def sample_feat(feat_map):
            sampled = torch.nn.functional.grid_sample(feat_map, grid, mode='bilinear', align_corners=False)
            return sampled.squeeze(2).transpose(1, 2)

        f_low_sampled = sample_feat(feat_low)   # [B*N, Q, 512]
        f_mid_sampled = sample_feat(feat_mid)   # [B*N, Q, 256]
        f_high_sampled = sample_feat(feat_high) # [B*N, Q, 128]
        
        # 3.2.3 InfiniDepth 门控层级融合！
        ms_patch_embeddings = self.ms_fusion(f_high_sampled, f_mid_sampled, f_low_sampled) # [B*N, Q, dec_embed_dim]

        # 3.3 Query Token embedding
        if is_sat_mask is not None:
            mask_tok = is_sat_mask.to(hidden.device).view(B, N, 1, 1)
            tok_sat_exp = self.query_token_sat.unsqueeze(0).expand(B, N, num_queries, -1)
            tok_grd_exp = self.query_token_grd.unsqueeze(0).expand(B, N, num_queries, -1)
            token_all = torch.where(mask_tok, tok_sat_exp, tok_grd_exp)
        else:
            token_sat = self.query_token_sat.unsqueeze(0).expand(B, 1, num_queries, -1)
            token_grd = self.query_token_grd.unsqueeze(0).expand(B, N-1, num_queries, -1)
            token_all = torch.cat([token_sat, token_grd], dim=1)

        token_all = token_all.reshape(B*N, num_queries, -1) # (B*N, Num_queries, dim)

        # 3.4 将位置编码、RGB patch embedding 和 Query token embedding 叠加，得到初始的 query_embeddings
        query_embeddings = query_pos_embeddings + token_all + ms_patch_embeddings  # (B*N, Num_queries, dim)
        ## 把hidden的patch_h*patch_w个patch token当作KV，送入 DecoderBlock 进行交叉注意力计算，得到 query_embeddings 的更新
        query_hidden = self.query_hidden_project(hidden[:, self.patch_start_idx:]) # (B*N, grid_hw, embed_dim)

        # 3.5 让 Query 去图像特征 (hidden) 中提取信息
        # --- RoPE 位置编码 ---
        # Q positions: queries (u=x, v=y) in [0,1] → (y, x) 缩放到 patch 网格坐标
        query_positions = torch.stack([
            queries[..., 1] * (patch_h - 1),   # v → y
            queries[..., 0] * (patch_w - 1),   # u → x
        ], dim=-1).detach()  # (B*N, Q, 2), no grad

        # KV positions: encoder patch tokens 在规则网格上的整数坐标
        ky = torch.arange(patch_h, device=hidden.device, dtype=hidden.dtype)
        kx = torch.arange(patch_w, device=hidden.device, dtype=hidden.dtype)
        kv_grid = torch.cartesian_prod(ky, kx)  # (patch_h*patch_w, 2), (y, x)
        kv_positions = kv_grid.unsqueeze(0).expand(B * N, -1, -1)  # (B*N, ph*pw, 2)

        for block in self.query_decoder:
            query_embeddings = block(
                query_embeddings, query_hidden,
                query_positions=query_positions, kv_positions=kv_positions
            )
        point_hidden = self.query_norm(query_embeddings)
        
        # 3.6. 处理相机hidden
        camera_hidden = self.camera_decoder(hidden, xpos=pos)
        if self.use_global_points:
            context = hidden.reshape(B, N, patch_h*patch_w+self.patch_start_idx, -1)[:, 0:1].repeat(1, N, 1, 1).reshape(B*N, patch_h*patch_w+self.patch_start_idx, -1)
            global_point_hidden = self.global_points_decoder(hidden, context, xpos=pos, ypos=pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # local points
            point_hidden = point_hidden.float()
            grd_ret = self.grd_point_head(point_hidden).reshape(B, N, query_per_view, -1)  # (B, N, Q, 3)
            sat_ret = self.sat_point_head(point_hidden).reshape(B, N, query_per_view, -1)   # (B, N, Q, 2)
            
            # --- 非卫星视图 (保持不变) ---
            xy, z = grd_ret.split([2, 1], dim=-1)
            z_pos = torch.exp(z)
            grd_points_all = torch.cat([xy * z_pos, z_pos], dim=-1)

            # --- 卫星视图 (正交缩放先验) ---
            # Plan A
            sat_log_mpp, sat_z = sat_ret.split([1, 1], dim=-1)  # 拆分出对数缩放系数和高度
            sat_z_pos = torch.exp(sat_z)
            # [核心逻辑 1] 保证同一张图 meter_per_pixel 唯一：在 Q 维度上做全局平均池化
            global_log_mpp = sat_log_mpp.mean(dim=2, keepdim=True) # (B, N, 1, 1)
            sat_mpp = MIN_MPP + (MAX_MPP - MIN_MPP) * torch.sigmoid(global_log_mpp) # 使用 exp 保证物理缩放系数必须为正数

            # [核心逻辑 2] 计算 XY：(U, V) - 0.5 是为了把相机原点定在图像正中心
            queries_view = queries.reshape(B, N, query_per_view, 2) # (B, N, Q, 2)
            wh = torch.tensor([W, H], dtype=sat_mpp.dtype, device=sat_mpp.device).view(1, 1, 1, 2)
            sat_xy = (queries_view - 0.5) * wh * sat_mpp # 精确的几何反投影
            sat_points_all = torch.cat([sat_xy, sat_z_pos], dim=-1)  # (B, N, Q, 3)                               # (B, N, Q, 3)
            
            # Plan B
            # sat_xy, sat_z = sat_ret.split([2, 1], dim=-1)
            # sat_points_all = torch.cat([sat_xy, sat_z], dim=-1)  # (B, N, Q, 3)

            mask = is_sat_mask.to(grd_ret.device).view(B, N, 1, 1)   # (B, N, 1, 1) bool
            local_points = torch.where(mask, sat_points_all, grd_points_all)

            # confidence
            if self.train_conf:
                conf = self.conf_head(point_hidden).reshape(B, N, query_per_view, -1)
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
