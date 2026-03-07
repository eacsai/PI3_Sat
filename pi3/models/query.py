import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CrossAttention(nn.Module):
    """Efficient cross-attention using PyTorch's scaled_dot_product_attention.

    Automatically uses FlashAttention or memory-efficient attention when available.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
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

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (B, N_q, C) query tokens
            key_value: (B, N_kv, C) key-value tokens (encoder features)
            mask: Optional attention mask

        Returns:
            out: (B, N_q, C)
        """
        B, N_q, C = query.shape
        N_kv = key_value.shape[1]

        # Project queries, keys, values
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Use PyTorch's efficient attention (FlashAttention when available)
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


class DecoderBlock(nn.Module):
    """Decoder block with cross-attention and MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads, qkv_bias, attn_drop, drop)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(
        self,
        query: torch.Tensor,
        encoder_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query: (B, N_q, C) query tokens
            encoder_features: (B, N_kv, C) encoder output (Global Scene Representation)

        Returns:
            out: (B, N_q, C)
        """
        # Cross-attention
        query = query + self.cross_attn(
            self.norm1(query),
            self.norm_kv(encoder_features)
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
        coords: torch.Tensor,
        t_src: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            frames: (B, T, C, H, W) video frames
            coords: (B, N, 2) normalized coordinates in [0, 1]
            t_src: (B, N) source frame indices

        Returns:
            embeddings: (B, N, embed_dim)
        """
        B, T, C, H, W = frames.shape
        N = coords.shape[1]
        ps = self.patch_size

        # Gather frames for each query
        # t_src: (B, N) -> expand for gathering
        t_src_expanded = t_src.view(B, N, 1, 1, 1).expand(-1, -1, C, H, W)

        # Create batch indices
        batch_frames = []
        for b in range(B):
            query_frames = frames[b, t_src[b]]  # (N, C, H, W)
            batch_frames.append(query_frames)
        query_frames = torch.stack(batch_frames, dim=0)  # (B, N, C, H, W)

        # Reshape for grid_sample: (B*N, C, H, W)
        query_frames = query_frames.view(B * N, C, H, W)

        # Create sampling grid
        # coords: (B, N, 2) -> pixel offsets
        coords_pixel = coords.clone()
        coords_pixel[..., 0] = coords_pixel[..., 0] * (W - 1)
        coords_pixel[..., 1] = coords_pixel[..., 1] * (H - 1)

        # Add offsets for patch: (B, N, ps, ps, 2)
        grid = coords_pixel.view(B, N, 1, 1, 2) + self.offsets.view(1, 1, ps, ps, 2)

        # Normalize to [-1, 1] for grid_sample
        grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0

        # Reshape grid: (B*N, ps, ps, 2)
        grid = grid.view(B * N, ps, ps, 2)

        # Sample patches
        patches = F.grid_sample(
            query_frames, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )  # (B*N, C, ps, ps)

        # Reshape and permute
        patches = patches.view(B, N, C, ps, ps)
        patches = patches.permute(0, 1, 3, 4, 2)  # (B, N, ps, ps, C)

        # Flatten and embed
        patches_flat = patches.reshape(B, N, -1)
        return self.mlp(patches_flat)