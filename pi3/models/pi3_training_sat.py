import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d, ContextTransformerDecoder
from .layers.camera_head import CameraHead
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file
from .sat_position import FourierEmbedder

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
            load_vggt=False,
            load_pi3=True,
            freeze_encoder=True,
            train_conf=False,
            num_dec_blk_not_to_checkpoint=4,
            ckpt=None,
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
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)
        # self.sat_point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=2)
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

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        self.train_conf = train_conf
        # 仅在训练 confidence 时创建分支；若 load_pi3=true 且 train_conf=false 也建 conf，
        # forward 从不调用它们 → DDP 报 unused parameters。
        if train_conf:
            assert ckpt is not None or load_pi3, "Please provide pi3 checkpoint to load confidence decoder."
            self.conf_decoder = deepcopy(self.point_decoder)
            self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)
        else:
            self.conf_decoder = None
            self.conf_head = None

        if train_conf:
            freeze_all_params([
                self.encoder, self.decoder,
                self.point_decoder, self.point_head,
                self.camera_decoder, self.camera_head,
                self.register_token, self.sat_pos_embedder,
            ])

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

    def decode(self, hidden, N, H, W, is_sat_mask, sat_pos_embed=None):
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
                is_sat_mask, # [B, N] bool
        ):  # [关键修改] 加入 queries 和 is_sat_mask 参数

        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14
        
        # encode by dinov2
        imgs = imgs.reshape(B*N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)

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
        hidden, pos = self.decode(hidden, N, H, W, is_sat_mask, sat_pos_embed)

        point_hidden = self.point_decoder(hidden, xpos=pos)
        if self.train_conf:
            conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            sat_xy = None
            # local points PlanA
            # point_hidden = point_hidden.float()
            # # LinearPts3d 需要：只输入 patch tokens（去掉 register token），并传入 img_shape
            # grd_points_all = self.point_head(
            #     [point_hidden[:, self.patch_start_idx:]], (H, W)
            # ).reshape(B, N, H, W, 3)
            # sat_ret = self.sat_point_head(
            #     [point_hidden[:, self.patch_start_idx:]], (H, W)
            # ).reshape(B, N, H, W, 2)

            # --- 卫星视图 (正交缩放先验) ---
            # sat_log_mpp, sat_z = sat_ret.split([1, 1], dim=-1)  # (B, N, H, W, 1)
            # # [核心逻辑 1] 保证同一张图 meter_per_pixel 唯一：对整幅图做全局平均
            # global_log_mpp = sat_log_mpp.mean(dim=(2, 3), keepdim=True)  # (B, N, 1, 1, 1)
            # sat_mpp = MIN_MPP + (MAX_MPP - MIN_MPP) * torch.sigmoid(global_log_mpp)  # 确保缩放系数为正

            # # [核心逻辑 2] 计算 XY：(U, V) - 0.5 是为了把相机原点定在图像正中心
            # x_idx = torch.arange(W, device=sat_ret.device, dtype=sat_mpp.dtype)
            # y_idx = torch.arange(H, device=sat_ret.device, dtype=sat_mpp.dtype)
            # # 像素中心坐标：u,v in (0,1)，减 0.5 后以图像中心为原点
            # u = (x_idx + 0.5) / W  # (W,)
            # v = (y_idx + 0.5) / H  # (H,)
            # grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')  # (H, W)
            # sat_grid = torch.stack([grid_u, grid_v], dim=-1)  # (H, W, 2)

            # wh = torch.tensor([W, H], device=sat_ret.device, dtype=sat_mpp.dtype)  # (2,)
            # sat_xy_base = (sat_grid - 0.5) * wh  # (H, W, 2), 单位：像素尺度
            # sat_xy = sat_xy_base[None, None, :, :, :] * sat_mpp  # (B, N, H, W, 2)
            # sat_points_all = torch.cat([sat_xy, sat_z], dim=-1)  # (B, N, H, W, 3)

            # mask = is_sat_mask.to(sat_ret.device).view(B, N, 1, 1, 1)  # (B, N, 1, 1, 1) bool
            # local_points = torch.where(mask, sat_points_all, grd_points_all)

            # local points PlanB
            point_hidden = point_hidden.float()
            local_points = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)

            # local points PlanC
            # point_hidden = point_hidden.float()
            # grd_points_all = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            # sat_xy, sat_z = grd_points_all.split([2, 1], dim=-1)
            # # 用 sat_xy 的 dtype/device，避免依赖 sat_mpp/sat_ret 在此分支是否已定义
            # x_idx = torch.arange(W, device=sat_xy.device, dtype=sat_xy.dtype)
            # y_idx = torch.arange(H, device=sat_xy.device, dtype=sat_xy.dtype)
            # # 像素中心坐标：u,v in (0,1)，减 0.5 后以图像中心为原点
            # u = (x_idx + 0.5) / W  # (W,)
            # v = (y_idx + 0.5) / H  # (H,)
            # grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')  # (H, W)
            # sat_grid = torch.stack([grid_u, grid_v], dim=-1)  # (H, W, 2)

            # wh = torch.tensor([W, H], device=sat_xy.device, dtype=sat_xy.dtype)  # (2,)
            # sat_xy_base = (sat_grid - 0.5) * wh  # (H, W, 2), 单位：像素尺度

            # # 估计单一标量 sat_mpp（x/y 共用）：最小二乘拟合 sat_xy ≈ sat_mpp * sat_xy_base
            # # 避免 sat_xy_base≈0 时直接相除带来的 Inf/NaN 放大
            # eps = 1e-6
            # sat_xy_base_ = sat_xy_base[None, None, :, :, :]  # (1,1,H,W,2)
            # num = (sat_xy * sat_xy_base_).sum(dim=(2, 3, 4), keepdim=True)  # (B,N,1,1,1)
            # den = (sat_xy_base_ ** 2).sum(dim=(2, 3, 4), keepdim=True).clamp_min(eps)  # (1,1,1,1,1)
            # sat_mpp = num / den  # (B,N,1,1,1)

            # sat_xy_new = sat_xy_base_ * sat_mpp  # (B,N,H,W,2)
            # sat_points_all = torch.cat([sat_xy_new, sat_z], dim=-1)
            # mask = is_sat_mask.to(sat_xy.device).view(B, N, 1, 1, 1)  # (B, N, 1, 1, 1) bool
            # local_points = torch.where(mask, sat_points_all, grd_points_all)

            # confidence
            if self.train_conf:
                conf_hidden = conf_hidden.float()
                conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            else:
                conf = None
                
            # camera
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)
            
            # unproject local points using camera poses
            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

        return dict(
            points=points,
            local_points=local_points,
            sat_ori_xy=sat_xy if sat_xy is not None else None,
            conf=conf,
            camera_poses=camera_poses,
            global_points=None
        )
