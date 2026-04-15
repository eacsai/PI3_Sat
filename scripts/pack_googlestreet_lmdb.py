"""Pack the googlestreet dataset into a compact, fast-read LMDB.

v2 key changes compared to the old packer:

1. One LMDB entry per *view* (not per file). The dataset previously had to
   `txn.get()` three to four files to reconstruct a single view (meta npy,
   depth npy or dap.pt, rgb jpg). Now each view is a single pickle blob that
   carries everything the dataloader needs.
2. Pre-resize RGB and depth at pack time to the training-time target sizes:
       ground / uav : 512 x 512
       satellite    : 1024 x 1024
   Depth is downscaled with nearest-neighbour (no interpolation across
   depth discontinuities); RGB uses cv2.INTER_AREA which is ideal for
   downscaling natural images.
3. Pre-scale camera intrinsics so the stored K matches the resized image
   exactly (see `_pack_view` for the satellite caveat).
4. Pre-resolve the ground-depth source — prefer `*_depth_dap.pt` when it
   is non-empty, otherwise fall back to `*_depth.npy` — so the runtime
   loader no longer needs the branch.
5. Pre-apply the dataset's depth clipping (ground > 60m → -1, uav > 300m
   → -1) so the stored depth is already in the "training-ready" form.
6. Parallelise the heavy lifting with a `multiprocessing.Pool`; the main
   process is the sole LMDB writer so there are no transaction races.
7. Skip `*_pano_*` views at pack time — the dataset never uses them.

LMDB layout produced:

    __VERSION__       → b"2"
    __DIR_CACHE__     → json{folder_path: {"p": [prefix, ...], "ry": float}}
    "<folder>|<prefix>" → pickle(dict(K, c2w, rgb uint8, depth float32, s))

Run:

    python scripts/pack_googlestreet_lmdb.py \
        --data-root /data/zhongyao/dataset \
        --lmdb-path /data/wangqw/dataset_lmdb_v2 \
        --workers 16
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import re
import shutil
import sys

import cv2
import lmdb
import numpy as np
import torch
from tqdm import tqdm

sys.path.append('.')
from datasets.googlestreet_dataset import get_sorted_pair_paths


VERSION = 2

# Training-time target resolutions.
GROUND_DRONE_TARGET = 512
SAT_TARGET = 1024

# Same clipping thresholds the dataset used to apply at runtime.
GROUND_DEPTH_CLIP = 60.0
UAV_DEPTH_CLIP = 300.0


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', s)]


def _view_type(prefix: str) -> str:
    if 'pano' in prefix:
        return 'pano'  # filtered out; dataset never uses these
    if 'satellite' in prefix:
        return 'satellite'
    if 'ground' in prefix:
        return 'ground'
    if 'uav' in prefix:
        return 'uav'
    return 'unknown'


def _load_depth(folder: str, prefix: str, view_type: str) -> np.ndarray:
    """Mirror the dataset's runtime depth-source selection. Always returns an
    owning float32 array that is safe to mutate in place."""
    if view_type == 'ground':
        dap_path = os.path.join(folder, f"{prefix}_depth_dap.pt")
        if os.path.isfile(dap_path):
            tensor = torch.load(dap_path, map_location='cpu', weights_only=True)
            if bool(tensor.any()):
                # Some stored tensors carry requires_grad=True — detach before .numpy().
                # .copy() detaches the numpy array from the torch storage so we can mutate.
                return tensor.detach().cpu().numpy().astype(np.float32).copy()
    return np.load(os.path.join(folder, f"{prefix}_depth.npy")).astype(np.float32)


def _resize_rgb(rgb: np.ndarray, target: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if h == target and w == target:
        return np.ascontiguousarray(rgb)
    return cv2.resize(rgb, (target, target), interpolation=cv2.INTER_AREA)


def _resize_depth_nearest(depth: np.ndarray, target: int) -> np.ndarray:
    h, w = depth.shape
    if h == target and w == target:
        return np.ascontiguousarray(depth)
    return cv2.resize(depth, (target, target), interpolation=cv2.INTER_NEAREST)


def _scale_K(K: np.ndarray, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> np.ndarray:
    K = np.asarray(K, dtype=np.float32).copy()
    sx = dst_hw[1] / src_hw[1]
    sy = dst_hw[0] / src_hw[0]
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return K


def _pack_view(args):
    """Worker: produce a single (key, value) for one view. Returns None on error."""
    folder, prefix = args
    try:
        view_type = _view_type(prefix)
        if view_type in ('unknown', 'pano'):
            return None
        is_sat = view_type == 'satellite'

        meta = np.load(os.path.join(folder, f"{prefix}_rgb.npy"), allow_pickle=True).item()
        K_src = np.asarray(meta['intrinsics'], dtype=np.float32)
        c2w = np.asarray(meta['c2w'], dtype=np.float32)

        rgb_file = f"{prefix}.jpg" if is_sat else f"{prefix}_rgb.jpg"
        rgb_bgr = cv2.imread(os.path.join(folder, rgb_file), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            return None
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        src_h, src_w = rgb.shape[:2]

        depth = _load_depth(folder, prefix, view_type)

        # Clip here (matching runtime semantics) so the dataset can skip the branch.
        if view_type == 'ground':
            depth = depth.copy()
            depth[depth > GROUND_DEPTH_CLIP] = -1
        elif view_type == 'uav':
            depth = depth.copy()
            depth[depth > UAV_DEPTH_CLIP] = -1

        target = SAT_TARGET if is_sat else GROUND_DRONE_TARGET
        rgb_small = _resize_rgb(rgb, target)
        depth_small = _resize_depth_nearest(depth, target)

        if is_sat:
            # Satellite K is calibrated for the *reference* 1024x1024 sat view
            # (principal point (512, 512)), while the raw jpeg was oversampled.
            # The runtime cropping code uses sat_W = current image width together
            # with SAT_RES to derive the effective focal length, and that formula
            # is invariant to sat_W — so resizing the RGB to SAT_TARGET does NOT
            # require rescaling the stored K.
            K_out = K_src.copy()
        else:
            # Ground / UAV K is calibrated for the raw resolution, scale to target.
            K_out = _scale_K(K_src, (src_h, src_w), (target, target))

        blob = {
            'K': K_out.astype(np.float32),
            'c2w': c2w.astype(np.float32),
            'rgb': np.ascontiguousarray(rgb_small, dtype=np.uint8),
            'depth': np.ascontiguousarray(depth_small, dtype=np.float32),
            's': bool(is_sat),
        }
        key = f"{folder}|{prefix}".encode('utf-8')
        value = pickle.dumps(blob, protocol=pickle.HIGHEST_PROTOCOL)
        return folder, prefix, float(c2w[1, 3]), key, value
    except Exception as e:  # noqa: BLE001 — best-effort; skip broken views
        print(f"[pack_view][error] {folder}|{prefix}: {type(e).__name__}: {e}", flush=True)
        return None


def _scan_folders(pair_paths: list[str]):
    """Collect per-folder prefixes and the reference ground prefix (first ground
    view, by natural sort). Folders without any ground view are dropped — the
    dataset requires one for its world-frame construction."""
    folder_prefixes: dict[str, list[str]] = {}
    ref_prefix_map: dict[str, str] = {}
    for folder in tqdm(pair_paths, desc="Scanning folders"):
        if not os.path.isdir(folder):
            continue
        try:
            entries = os.listdir(folder)
        except OSError:
            continue
        prefixes = sorted(
            p for p in (e[:-len('_rgb.npy')] for e in entries if e.endswith('_rgb.npy'))
            if _view_type(p) not in ('unknown', 'pano')
        )
        if not prefixes:
            continue
        ground_prefixes = sorted(
            [p for p in prefixes if _view_type(p) == 'ground'],
            key=_natural_key,
        )
        if not ground_prefixes:
            continue
        folder_prefixes[folder] = prefixes
        ref_prefix_map[folder] = ground_prefixes[0]
    return folder_prefixes, ref_prefix_map


def pack_to_lmdb(
    data_root: str,
    lmdb_path: str,
    workers: int,
    commit_every: int,
) -> None:
    print(
        f"Packing {data_root} into LMDB at {lmdb_path} (v{VERSION}) with {workers} workers",
        flush=True,
    )
    if not os.path.exists(data_root):
        print(f"Directory {data_root} does not exist.")
        return

    pair_paths_train = get_sorted_pair_paths(data_root, split=False, mode='train')
    pair_paths_test = get_sorted_pair_paths(data_root, split=False, mode='test')
    pair_paths = sorted(set(pair_paths_train + pair_paths_test))

    if os.path.exists(lmdb_path):
        print(f"Cleaning stale LMDB at {lmdb_path}...")
        shutil.rmtree(lmdb_path)
    os.makedirs(lmdb_path, exist_ok=True)

    folder_prefixes, ref_prefix_map = _scan_folders(pair_paths)

    tasks = [
        (folder, prefix)
        for folder, prefixes in folder_prefixes.items()
        for prefix in prefixes
    ]
    print(
        f"Total folders: {len(folder_prefixes)}, total views: {len(tasks)}",
        flush=True,
    )

    # 1 TiB virtual cap is plenty for ~40k folders * ~21 MB/folder ≈ 840 GB.
    # The file is sparse (only the written bytes consume disk), so this is only
    # a safety ceiling — bump it if you ever pack a bigger corpus.
    map_size = 1099511627776
    env = lmdb.open(
        lmdb_path,
        map_size=map_size,
        writemap=True,
        map_async=True,
        sync=False,
        meminit=False,
    )

    txn = env.begin(write=True)
    written = 0
    ref_ground_y: dict[str, float] = {}

    ctx = mp.get_context('fork')
    with ctx.Pool(processes=workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_pack_view, tasks, chunksize=8),
            total=len(tasks),
            desc='Packing views',
        ):
            if result is None:
                continue
            folder, prefix, c2w_y, key, value = result
            txn.put(key, value)
            written += 1
            if ref_prefix_map.get(folder) == prefix:
                ref_ground_y[folder] = c2w_y
            if written % commit_every == 0:
                txn.commit()
                txn = env.begin(write=True)
    txn.commit()

    dir_cache: dict[str, dict] = {}
    for folder, prefixes in folder_prefixes.items():
        if folder not in ref_ground_y:
            continue
        dir_cache[folder] = {
            'p': prefixes,
            'ry': ref_ground_y[folder],
        }

    with env.begin(write=True) as txn:
        txn.put(b'__VERSION__', str(VERSION).encode('utf-8'))
        txn.put(b'__DIR_CACHE__', json.dumps(dir_cache).encode('utf-8'))

    env.sync()
    env.close()
    print(
        f"Done! LMDB saved at {lmdb_path} — {written} views, {len(dir_cache)} folders",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='/data/zhongyao/dataset')
    parser.add_argument('--lmdb-path', default='/data/wangqw/dataset_lmdb_v2')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument('--commit-every', type=int, default=256)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    pack_to_lmdb(
        data_root=args.data_root,
        lmdb_path=args.lmdb_path,
        workers=args.workers,
        commit_every=args.commit_every,
    )
