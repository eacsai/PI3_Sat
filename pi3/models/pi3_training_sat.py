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
            use_global_points=False,
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
        
        self.sat_to_embed = nn.Linear(3, self.encoder.embed_dim)
        nn.init.constant_(self.sat_to_embed.weight, 0)
        nn.init.constant_(self.sat_to_embed.bias, 0)
        self.register_buffer('sat_physical_range', torch.tensor(140.0))

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

        self.sat_point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        self.sat_point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

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

        if load_vggt:
            vggt_weight = load_file('ckpts/VGGT-1B/model.safetensors')
            vggt_enc_weight = {k.replace('aggregator.patch_embed.', ''):vggt_weight[k] for k in list(vggt_weight.keys()) if k.startswith('aggregator.patch_embed.')}
            print("Loading vggt encoder", self.encoder.load_state_dict(vggt_enc_weight, strict=False))

            vggt_dec_weight = {k.replace('aggregator.global_blocks.', ''):vggt_weight[k] for k in list(vggt_weight.keys()) if k.startswith('aggregator.global_blocks.')}
            vggt_dec_weight1 = {}
            for k in list(vggt_dec_weight.keys()):
                idx = k.split('.')[0]
                other = k[len(idx):]
                vggt_dec_weight1[f'{int(idx)*2 + 1}{other}'] = vggt_dec_weight[k]
            vggt_dec_weight = vggt_dec_weight1 

            vggt_dec_weight_frame = {k.replace('aggregator.frame_blocks.', ''):vggt_weight[k] for k in list(vggt_weight.keys()) if k.startswith('aggregator.frame_blocks.')}
            for k in list(vggt_dec_weight_frame.keys()):
                idx = k.split('.')[0]
                other = k[len(idx):]
                vggt_dec_weight[f'{int(idx)*2}{other}'] = vggt_dec_weight_frame[k]

            print("Loading vggt decoder", self.decoder.load_state_dict(vggt_dec_weight, strict=False))

        self.train_conf = train_conf
        # ----------------------
        #     Conf Decoder
        # ----------------------
        self.conf_decoder = deepcopy(self.point_decoder)
        self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)
        self.sat_conf_decoder = deepcopy(self.sat_point_decoder)
        self.sat_conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)

        if train_conf:
            freeze_all_params([self.encoder, self.decoder, self.point_decoder, self.point_head, self.sat_point_decoder, self.sat_point_head, self.camera_decoder,  self.camera_head, self.register_token])
        if use_global_points:
            freeze_all_params([self.global_points_decoder, self.global_point_head])

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

            # -------------------------------------------------------
            # 2.  [新增] 手动克隆权重给 Satellite 模块
            # -------------------------------------------------------
            new_keys = {}
            for k, v in pi3_weight.items():
                # 1. 复制 point_decoder -> sat_point_decoder
                if k.startswith('point_decoder.'):
                    new_k = k.replace('point_decoder.', 'sat_point_decoder.')
                    new_keys[new_k] = v
                
                # 2. 复制 point_head -> sat_point_head
                elif k.startswith('point_head.'):
                    new_k = k.replace('point_head.', 'sat_point_head.')
                    new_keys[new_k] = v
                
                # 3. 复制 conf_decoder -> sat_conf_decoder (如果有的话)
                elif k.startswith('conf_decoder.'):
                    new_k = k.replace('conf_decoder.', 'sat_conf_decoder.')
                    new_keys[new_k] = v
                
                # 4. 复制 conf_head -> sat_conf_head (如果有的话)
                elif k.startswith('conf_head.'):
                    new_k = k.replace('conf_head.', 'sat_conf_head.')
                    new_keys[new_k] = v
            
            # 将复制出来的新权重合并回原来的字典中
            if len(new_keys) > 0:
                print(f"[Pi3] Duplicating {len(new_keys)} keys for satellite components initialization.")
                pi3_weight.update(new_keys)
            # -------------------------------------------------------

            print("Loading pi3 weights", self.load_state_dict(pi3_weight, strict=False))

    def decode(self, hidden, N, H, W):
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
    
    def forward(self, imgs):
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape

        patch_h, patch_w = H // 14, W // 14
        
        # encode by dinov2
        imgs = imgs.reshape(B*N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        # ==========================================================
        # [新增] 在这里注入卫星位置编码
        # ==========================================================
        # hidden 目前是 (B*N, L, D)，我们需要把它拆开成 (B, N, L, D)
        hidden = hidden.reshape(B, N, patch_h * patch_w, -1)
        # 取出卫星图特征 (View 0)
        sat_feat = hidden[:, 0] # (B, L, D)
        
        # 准备位置编码
        y_steps = torch.linspace(-1, 1, patch_h, device=sat_feat.device, dtype=sat_feat.dtype)
        x_steps = torch.linspace(-1, 1, patch_w, device=sat_feat.device, dtype=sat_feat.dtype)
        grid_y, grid_x = torch.meshgrid(y_steps, x_steps, indexing='ij')
        scale_map = torch.full_like(grid_x, self.sat_physical_range)
        pos_input = torch.stack([grid_x, grid_y, scale_map], dim=-1)
        pos_embed = self.sat_to_embed(pos_input)
        pos_embed = pos_embed.unsqueeze(0).expand(B, -1, -1, -1)

        # 这就是注入过程。卫星图的特征现在包含了特定的位置信息。
        sat_feat = sat_feat + pos_embed.reshape(B, patch_h*patch_w, -1)
        
        # 放回去并恢复形状给 self.decode 使用
        hidden[:, 0] = sat_feat
        hidden = hidden.reshape(B*N, patch_h * patch_w, -1)

        hidden, pos = self.decode(hidden, N, H, W)

        BN_hidden = hidden.reshape(B, N, patch_h*patch_w+self.patch_start_idx, -1)
        BN_pos = pos.reshape(B, N, patch_h*patch_w+self.patch_start_idx, -1)
        
        # 2. 拆分 Satellite (View 0) 和 Ground (View 1:)
        sat_hidden = BN_hidden[:, 0]     # (B, tokens, dim)
        sat_pos = BN_pos[:, 0]
        
        gd_hidden = BN_hidden[:, 1:]     # (B, N-1, tokens, dim)
        gd_pos = BN_pos[:, 1:]

        sat_point_hidden = self.sat_point_decoder(sat_hidden, xpos=sat_pos) # (B, tokens, dim)
        gd_hidden_flat = gd_hidden.reshape(B*(N-1), patch_h*patch_w+self.patch_start_idx, -1) # (B*(N-1), tokens, dim)
        gd_pos_flat = gd_pos.reshape(B*(N-1), patch_h*patch_w+self.patch_start_idx, -1) # (B*(N-1), tokens, dim)
        gd_point_hidden = self.point_decoder(gd_hidden_flat, xpos=gd_pos_flat) # (B*(N-1), tokens, dim)

        if self.train_conf:
            sat_conf_hidden = self.sat_conf_decoder(sat_hidden, xpos=sat_pos) # (B, tokens, dim)
            gd_conf_hidden = self.conf_decoder(gd_hidden_flat, xpos=gd_pos_flat) # (B*(N-1), tokens, dim)

        camera_hidden = self.camera_decoder(hidden, xpos=pos)
        if self.use_global_points:
            context = hidden.reshape(B, N, patch_h*patch_w+self.patch_start_idx, -1)[:, 0:1].repeat(1, N, 1, 1).reshape(B*N, patch_h*patch_w+self.patch_start_idx, -1)
            global_point_hidden = self.global_points_decoder(hidden, context, xpos=pos, ypos=pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # TODO: 对于卫星图，不需要预测深度的对数，也不要把xy*z，而是直接预测x,y,z
            # sat local points
            sat_point_hidden = sat_point_hidden.float()
            sat_local_points = self.sat_point_head([sat_point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, 1, H, W, -1)
            # grd and drone local points
            gd_point_hidden = gd_point_hidden.float()
            gd_ret = self.point_head([gd_point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N-1, H, W, -1)
            gd_xy, gd_z = gd_ret.split([2, 1], dim=-1)
            gd_z = torch.exp(gd_z)
            gd_local_points = torch.cat([gd_xy * gd_z, gd_z], dim=-1)
            local_points = torch.cat([sat_local_points, gd_local_points], dim=1)
            # confidence
            if self.train_conf:
                sat_conf_hidden = sat_conf_hidden.float()
                sat_conf = self.sat_conf_head([sat_conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, 1, H, W, -1)
                gd_conf_hidden = gd_conf_hidden.float()
                gd_conf = self.conf_head([gd_conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N-1, H, W, -1)
                conf = torch.cat([sat_conf, gd_conf], dim=1)
            else:
                conf = None
                
            # camera
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)

            # Global points
            if self.use_global_points:
                global_point_hidden = global_point_hidden.float()
                global_points = self.global_point_head([global_point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            else:
                global_points = None
            
            # unproject local points using camera poses
            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

        return dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
            global_points=global_points
        )
