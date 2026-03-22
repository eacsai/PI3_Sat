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
    # TODO: 这里valid_indices是否合理
    valid_indices = np.flatnonzero(valid_mask)
    if valid_indices.size < Q:
        raise ValueError(f"Not enough valid pixels ({valid_indices.size} < {Q})")

    k_edge = min(int(Q * edge_ratio), valid_indices.size)
    k_rand = Q - k_edge

    # Sobel 梯度（只在 valid 像素上取值）
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    grad_at_valid = np.hypot(gx.ravel()[valid_indices], gy.ravel()[valid_indices])

    # argpartition 选出梯度最大的 k_edge 个（O(n) 复杂度）
    edge_local = np.argpartition(-grad_at_valid, k_edge)[:k_edge]

    # 从剩余 valid 像素中随机抽取 k_rand 个
    remain_local = np.delete(np.arange(valid_indices.size), edge_local)
    rand_local = remain_local[rng.choice(remain_local.size, size=k_rand, replace=False)]

    sampled = valid_indices[np.concatenate([edge_local, rand_local])]

    ys, xs = np.divmod(sampled, W)
    return np.stack([(xs + 0.5) / W, (ys + 0.5) / H], axis=-1).astype(np.float32)