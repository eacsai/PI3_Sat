import numpy as np
import cv2


def sample_query_uv(depth, valid_mask, rng, Q=8192, edge_ratio=0.3):
    """向量化地从深度图中采样 Q 个 query 点，偏重深度边缘，只在有效区域采样。

    depth: (H, W)
    valid_mask: (H, W)
    rng: np.random.Generator
    Q: int
    edge_ratio: float

    若有效像素不足 Q 个，抛出 ValueError 让上层换数据。
    返回 (Q, 2) 的 float32 数组，值域 [0, 1]。
    """
    H, W = depth.shape
    valid_indices = np.flatnonzero(valid_mask)
    if valid_indices.size < Q:
        raise ValueError(f"Not enough valid pixels ({valid_indices.size} < {Q})")

    k_edge = min(int(Q * edge_ratio), valid_indices.size)
    k_rand = Q - k_edge

    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    grad_at_valid = np.hypot(gx.ravel()[valid_indices], gy.ravel()[valid_indices])

    edge_local = np.argpartition(-grad_at_valid, k_edge)[:k_edge]

    remain_local = np.delete(np.arange(valid_indices.size), edge_local)
    rand_local = remain_local[rng.choice(remain_local.size, size=k_rand, replace=False)]

    sampled = valid_indices[np.concatenate([edge_local, rand_local])]

    ys, xs = np.divmod(sampled, W)
    return np.stack([(xs + 0.5) / W, (ys + 0.5) / H], axis=-1).astype(np.float32)


def sample_query_uv_with_patches(depth, valid_mask, rng, Q=8192, patch_size=3, edge_ratio=0.3):
    """微面片采样：将 query 组织成 patch_size×patch_size 的小面片 + 剩余随机点。

    返回的 UV 数组中，前 n_patches * patch_size² 个点按连续分组排列，
    每组 patch_size² 个点构成一个面片（行优先排列），可直接 reshape
    为 (n_patches, patch_size, patch_size, 2)。其余为散布的随机采样点。

    Returns:
        query_uv : (Q, 2) float32, UV 坐标 ∈ [0, 1]
        n_patches: int, 完整面片的数量
    """
    H, W = depth.shape
    half = patch_size // 2
    ps2 = patch_size * patch_size

    # ---- 1. 腐蚀 valid_mask，得到"面片中心候选"区域 ----
    kernel = np.ones((patch_size, patch_size), np.uint8)
    center_valid = cv2.erode(valid_mask.astype(np.uint8), kernel, iterations=1) > 0
    center_indices = np.flatnonzero(center_valid)

    n_patches_want = Q // ps2
    if center_indices.size < n_patches_want:
        n_patches_want = center_indices.size
    n_random = Q - n_patches_want * ps2

    all_valid = np.flatnonzero(valid_mask)
    if all_valid.size < n_random + n_patches_want:
        raise ValueError(
            f"Not enough valid pixels for patch sampling "
            f"({all_valid.size} < {n_random + n_patches_want})"
        )

    uvs_parts = []

    # ---- 2. 面片中心采样（边缘优先） ----
    if n_patches_want > 0:
        gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.hypot(gx.ravel()[center_indices], gy.ravel()[center_indices])

        k_edge = min(int(n_patches_want * edge_ratio), center_indices.size)
        k_rand_c = n_patches_want - k_edge

        if 0 < k_edge < center_indices.size:
            top_edge = np.argpartition(-grad, k_edge)[:k_edge]
        else:
            top_edge = np.arange(min(k_edge, center_indices.size))

        remain = np.delete(np.arange(center_indices.size), top_edge)
        if k_rand_c > 0 and remain.size > 0:
            chosen = remain[rng.choice(remain.size, size=min(k_rand_c, remain.size), replace=False)]
        else:
            chosen = np.array([], dtype=int)

        sel = np.concatenate([top_edge, chosen])[:n_patches_want]
        selected = center_indices[sel]
        cy, cx = np.divmod(selected, W)

        # 3×3 偏移量（行优先）
        dy_grid, dx_grid = np.meshgrid(
            np.arange(-half, half + 1),
            np.arange(-half, half + 1),
            indexing='ij',
        )
        dy_flat = dy_grid.ravel()  # (ps2,)
        dx_flat = dx_grid.ravel()

        # 向量化展开每个面片的所有点 → (n_patches_want, ps2)
        px = cx[:, None] + dx_flat[None, :]
        py = cy[:, None] + dy_flat[None, :]

        pu = (px + 0.5) / W
        pv = (py + 0.5) / H
        patch_uv = np.stack([pu, pv], axis=-1).reshape(-1, 2)  # (n_patches_want * ps2, 2)
        uvs_parts.append(patch_uv)

    # ---- 3. 剩余随机点 ----
    if n_random > 0:
        rand_idx = all_valid[rng.choice(all_valid.size, size=n_random, replace=False)]
        ry, rx = np.divmod(rand_idx, W)
        rand_uv = np.stack([(rx + 0.5) / W, (ry + 0.5) / H], axis=-1)
        uvs_parts.append(rand_uv)

    query_uv = np.concatenate(uvs_parts, axis=0).astype(np.float32)

    # 确保恰好 Q 个点
    if query_uv.shape[0] > Q:
        query_uv = query_uv[:Q]
    elif query_uv.shape[0] < Q:
        deficit = Q - query_uv.shape[0]
        pad_idx = all_valid[rng.choice(all_valid.size, size=deficit, replace=True)]
        pad_y, pad_x = np.divmod(pad_idx, W)
        pad_uv = np.stack([(pad_x + 0.5) / W, (pad_y + 0.5) / H], axis=-1).astype(np.float32)
        query_uv = np.concatenate([query_uv, pad_uv], axis=0)

    return query_uv, int(n_patches_want)