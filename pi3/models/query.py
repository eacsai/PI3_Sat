import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ContinuousRoPE2D(nn.Module):
    """RoPE2D for continuous (float) 2D positions.

    Standard RoPE2D (pos_embed.py) requires integer grid positions via F.embedding.
    This variant computes rotary embeddings directly from continuous (y, x)
    coordinates, making it suitable for query points at arbitrary positions.

    Interface matches RoPE2D: forward(tokens, positions)
        tokens:    (B, num_heads, N, head_dim)
        positions: (B, N, 2)  — continuous (y, x)
    """

    def __init__(self, freq: float = 100.0):
        super().__init__()
        self.base = freq

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        assert tokens.size(3) % 2 == 0
        D = tokens.size(3) // 2

        inv_freq = 1.0 / (self.base ** (
            torch.arange(0, D, 2, device=tokens.device).float() / D
        ))

        pos_y = positions[..., 0].float()
        pos_x = positions[..., 1].float()

        freqs_y = torch.einsum('bn,d->bnd', pos_y, inv_freq)
        freqs_x = torch.einsum('bn,d->bnd', pos_x, inv_freq)

        freqs_y = torch.cat([freqs_y, freqs_y], dim=-1).to(tokens.dtype)
        freqs_x = torch.cat([freqs_x, freqs_x], dim=-1).to(tokens.dtype)

        cos_y = freqs_y.cos().unsqueeze(1)
        sin_y = freqs_y.sin().unsqueeze(1)
        cos_x = freqs_x.cos().unsqueeze(1)
        sin_x = freqs_x.sin().unsqueeze(1)

        t_y, t_x = tokens.chunk(2, dim=-1)
        t_y = t_y * cos_y + ContinuousRoPE2D.rotate_half(t_y) * sin_y
        t_x = t_x * cos_x + ContinuousRoPE2D.rotate_half(t_x) * sin_x

        return torch.cat([t_y, t_x], dim=-1)


class CrossAttention(nn.Module):
    """Efficient cross-attention using PyTorch's scaled_dot_product_attention.

    Automatically uses FlashAttention or memory-efficient attention when available.
    Supports optional ContinuousRoPE2D with separate Q/K positions.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope: Optional[ContinuousRoPE2D] = None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        q_positions: Optional[torch.Tensor] = None,
        kv_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (B, N_q, C) query tokens
            key_value: (B, N_kv, C) key-value tokens (encoder features)
            mask: Optional attention mask
            q_positions: (B, N_q, 2) optional continuous 2D positions (y, x) for Q
            kv_positions: (B, N_kv, 2) optional 2D positions (y, x) for K

        Returns:
            out: (B, N_q, C)
        """
        B, N_q, C = query.shape
        N_kv = key_value.shape[1]

        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        if self.rope is not None and q_positions is not None and kv_positions is not None:
            q = self.rope(q, q_positions)
            k = self.rope(k, kv_positions)

        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.attn_drop if self.training else 0.0
        )

        x = x.transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """MLP block."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0
    ):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SelfAttention(nn.Module):
    """Self-attention with fused QKV projection, optional RoPE, using FlashAttention when available."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope: Optional[ContinuousRoPE2D] = None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) input tokens
            positions: (B, N, 2) optional continuous 2D positions (y, x) for RoPE
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.rope is not None and positions is not None:
            q = self.rope(q, positions)
            k = self.rope(k, positions)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DecoderBlock(nn.Module):
    """Decoder block: (optional) self-attention (w/ RoPE) → cross-attention → MLP.

    Self-attention enforces spatial coherence among query predictions.
    ContinuousRoPE2D encodes 2D query positions into the self-attention,
    so nearby queries attend to each other more strongly.
    Output projection is zero-initialized for smooth fine-tuning from
    pretrained weights that were trained without self-attention.
    """

    SA_CHUNK = 16384

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        rope: Optional[ContinuousRoPE2D] = None, # 这里的 ContinuousRoPE2D 替换为你的 RoPE 类型
        use_self_attn: bool = True  # ✨ 新增参数：默认开启 self-attention
    ):
        super().__init__()
        self.use_self_attn = use_self_attn

        # ✨ 根据参数决定是否初始化 Self-attention
        if self.use_self_attn:
            self.norm_self = nn.LayerNorm(dim)
            self.self_attn = SelfAttention(dim, num_heads, qkv_bias, attn_drop, drop, rope=rope)
            
            # Zero-init self-attn output so pretrained cross-attn weights stay effective
            nn.init.zeros_(self.self_attn.proj.weight)
            nn.init.zeros_(self.self_attn.proj.bias)
        else:
            # 保持属性存在，但设为 None
            self.norm_self = None
            self.self_attn = None

        # Cross-attention: query → encoder features (with optional RoPE)
        self.norm1 = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads, qkv_bias, attn_drop, drop, rope=rope)

        # FFN
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(
        self,
        query: torch.Tensor,
        encoder_features: torch.Tensor,
        query_positions: Optional[torch.Tensor] = None,
        kv_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (B, N_q, C) query tokens
            encoder_features: (B, N_kv, C) encoder output (Global Scene Representation)
            query_positions: (B, N_q, 2) continuous 2D positions (y, x) for Q in self/cross-attn
            kv_positions: (B, N_kv, 2) 2D grid positions (y, x) for K in cross-attn

        Returns:
            out: (B, N_q, C)
        """
        
        # ✨ 只有开启了 use_self_attn 才会执行这段逻辑
        if self.use_self_attn:
            # Self-attention with RoPE (chunked when Q is very large)
            N_q = query.shape[1]
            if N_q <= self.SA_CHUNK:
                query = query + self.self_attn(self.norm_self(query), positions=query_positions)
            else:
                normed = self.norm_self(query)
                chunks = normed.split(self.SA_CHUNK, dim=1)
                if query_positions is not None:
                    pos_chunks = query_positions.split(self.SA_CHUNK, dim=1)
                    sa_out = torch.cat(
                        [self.self_attn(c, positions=p) for c, p in zip(chunks, pos_chunks)],
                        dim=1
                    )
                else:
                    sa_out = torch.cat(
                        [self.self_attn(chunk) for chunk in chunks],
                        dim=1
                    )
                query = query + sa_out

        # Cross-attention with RoPE (Q at query positions, K at encoder grid positions)
        query = query + self.cross_attn(
            self.norm1(query),
            self.norm_kv(encoder_features),
            q_positions=query_positions,
            kv_positions=kv_positions
        )
        
        # MLP
        query = query + self.mlp(self.norm2(query))

        return query
    
class PatchEmbedding(nn.Module):
    """Local RGB patch embedding.

    Extracts and embeds a local patch around each query point.
    This dramatically improves performance by providing low-level appearance cues.
    """

    def __init__(self, patch_size: int = 9, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # MLP to embed flattened RGB patch
        patch_dim = patch_size * patch_size * 3
        self.mlp = nn.Sequential(
            nn.Linear(patch_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def extract_patches(
        self,
        frames: torch.Tensor,
        coords: torch.Tensor,
        t_src: torch.Tensor
    ) -> torch.Tensor:
        """Extract local patches around query coordinates.

        Args:
            frames: (B, T, C, H, W) video frames
            coords: (B, N, 2) normalized coordinates in [0, 1]
            t_src: (B, N) source frame indices

        Returns:
            patches: (B, N, patch_size, patch_size, 3)
        """
        B, T, C, H, W = frames.shape
        N = coords.shape[1]
        device = frames.device

        # Denormalize coordinates to pixel space
        u = coords[..., 0] * (W - 1)  # (B, N)
        v = coords[..., 1] * (H - 1)  # (B, N)

        # Get integer coordinates (center of patch)
        u_int = u.long()
        v_int = v.long()

        half_size = self.patch_size // 2
        patches = []

        for b in range(B):
            batch_patches = []
            for n in range(N):
                t = t_src[b, n].item()
                cx = u_int[b, n].item()
                cy = v_int[b, n].item()

                # Extract patch with padding for boundary cases
                frame = frames[b, t]  # (C, H, W)

                # Compute patch boundaries with clamping
                x_start = max(0, cx - half_size)
                x_end = min(W, cx + half_size + 1)
                y_start = max(0, cy - half_size)
                y_end = min(H, cy + half_size + 1)

                # Extract patch
                patch = frame[:, y_start:y_end, x_start:x_end]  # (C, h, w)

                # Pad if necessary
                pad_left = half_size - (cx - x_start)
                pad_right = half_size - (x_end - cx - 1)
                pad_top = half_size - (cy - y_start)
                pad_bottom = half_size - (y_end - cy - 1)

                if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                    patch = F.pad(patch, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')

                batch_patches.append(patch)

            patches.append(torch.stack(batch_patches, dim=0))  # (N, C, ps, ps)

        patches = torch.stack(patches, dim=0)  # (B, N, C, ps, ps)
        patches = patches.permute(0, 1, 3, 4, 2)  # (B, N, ps, ps, C)

        return patches

    def forward(
        self,
        frames: torch.Tensor,
        coords: torch.Tensor,
        t_src: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            frames: (B, T, C, H, W) video frames
            coords: (B, N, 2) normalized coordinates
            t_src: (B, N) source frame indices

        Returns:
            embeddings: (B, N, embed_dim)
        """
        patches = self.extract_patches(frames, coords, t_src)
        B, N = patches.shape[:2]

        # Flatten patches
        patches_flat = patches.reshape(B, N, -1)  # (B, N, ps*ps*3)

        # Embed
        return self.mlp(patches_flat)


class PatchEmbeddingFast(nn.Module):
    """Faster vectorized patch embedding using grid_sample."""

    def __init__(self, patch_size: int = 9, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        patch_dim = patch_size * patch_size * 3
        self.mlp = nn.Sequential(
            nn.Linear(patch_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Create relative grid offsets
        half = patch_size // 2
        offsets = torch.stack(torch.meshgrid(
            torch.arange(-half, half + 1),
            torch.arange(-half, half + 1),
            indexing='xy'
        ), dim=-1).float()  # (ps, ps, 2)
        self.register_buffer('offsets', offsets)

    def forward(
        self,
        frames: torch.Tensor,
        coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            frames: (B, T, C, H, W) video frames
            coords: (B*T, N, 2) normalized coordinates in [0, 1]

        Returns:
            embeddings: (B*T, N, embed_dim)
        """
        B, T, C, H, W = frames.shape
        frames = frames.reshape(B * T, C, H, W)
        N = coords.shape[1]
        ps = self.patch_size

        # 1. 直接计算像素网格坐标
        coords_pixel = coords.clone()
        coords_pixel[..., 0] = coords_pixel[..., 0] * (W - 1)
        coords_pixel[..., 1] = coords_pixel[..., 1] * (H - 1)

        # 2. 加上偏移量: (B*T, N, 1, 1, 2) + (1, 1, ps, ps, 2) -> (B*T, N, ps, ps, 2)
        grid = coords_pixel.view(B * T, N, 1, 1, 2) + self.offsets.view(1, 1, ps, ps, 2)

        # 3. 归一化到 [-1, 1]
        grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0

        # 【核心修正】：巧妙变形 grid 而不复制 frames！
        # 将 grid 变为 (B_T, N * ps, ps, 2)，这样 grid_sample 会把它当成一张极高的"瘦长"图来采
        grid = grid.view(B * T, N * ps, ps, 2)

        # 4. 执行极速采样 (完全不占用额外显存)
        patches = F.grid_sample(
            frames, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )  # 结果为 (B*T, C, N * ps, ps)

        # 5. 变回我们需要的形状
        patches = patches.view(B * T, C, N, ps, ps)
        patches = patches.permute(0, 2, 3, 4, 1)  # (B*T, N, ps, ps, C)

        # 6. 展平和 MLP 编码
        patches_flat = patches.reshape(B * T, N, -1)
        return self.mlp(patches_flat)