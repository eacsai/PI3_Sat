import sys
sys.path.append('.')

from datasets.base.base_dataset import BaseDataset
import os
import pickle
import numpy as np
from PIL import Image
from datasets.base.transforms import *
import json
import re
import cv2  # 新增：用于在 CPU 上极速处理深度图插值

SAT_RES = 500 # 卫星图分辨率(米)
SAT_METERS = 140  # 卫星图覆盖的实际范围(米)

def get_sort_priority(filename):
    # 注意顺序：必须先判断 satellite，因为 satellite 文件名中也包含 ground
    if 'satellite' in filename:
        return 0  # 优先级最高
    elif 'ground' in filename:
        return 1  # 优先级中等
    elif 'uav' in filename:
        return 2  # 优先级最低
    return 3      # 其他情况

def get_sorted_pair_paths(root_dir='.', split=True, mode='train'):
    """
    遍历指定目录结构，收集最底层的 pair_xxx 文件夹路径并排序。
    """
    target_paths = []

    if not os.path.exists(root_dir):
        print(f"错误: 目录 '{root_dir}' 不存在")
        return []

    level1_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    
    if mode == 'train':
        target_suffixes = ('0001_pair', '0004_pair', '0005_pair', '0007_pair', '0008_pair', '0012_pair', '0016_pair', '0017_pair', '0022_pair', '0023_pair', '0025_pair', '0027_pair', '0032_pair', '0035_pair', '0036_pair', '0056_pair', '0057_pair')
        # exclude_suffixes = (
        #     '0013_pair', '0516_pair', '0515_pair', '0512_pair', '0508_pair', 
        #     ' 0507_pair', '0506_pair', '0505_pair', '0503_pair', '0502_pair', 
        #     '0501_pair', '0496_pair', ' 0493_pair', '0472_pair', '0455_pair',
        #     '0446_pair', '0411_pair', ' 0407_pair', '0377_pair', '0360_pair',
        # )
        exclude_suffixes = ('0516_pair')
    else:
        target_suffixes = ('0516_pair',)

    for l1_name in level1_names:
        if mode == 'train':
            if not l1_name.endswith('_pair') or l1_name in exclude_suffixes:
                continue
        else:
            if not l1_name.endswith(target_suffixes):
                continue

        # if not l1_name.endswith(target_suffixes):
        #     continue

        l1_path = os.path.join(root_dir, l1_name)
        level2_names = sorted([d for d in os.listdir(l1_path) if os.path.isdir(os.path.join(l1_path, d))])
        
        for l2_name in level2_names:
            l2_path = os.path.join(l1_path, l2_name)
            level3_names = [d for d in os.listdir(l2_path) if os.path.isdir(os.path.join(l2_path, d))]
            
            for l3_name in level3_names:
                if l3_name.startswith('pair_'):
                    full_path = os.path.join(l2_path, l3_name)
                    target_paths.append(full_path)

    def natural_key(string):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', string)]

    target_paths.sort(key=natural_key)
    
    if split:
        paths_1 = [os.path.join(p, '_1') for p in target_paths]
        paths_2 = [os.path.join(p, '_2') for p in target_paths]
        final_paths = paths_1 + paths_2
    else:
        final_paths = target_paths

    return final_paths


class GoogleStreetDataset(BaseDataset):
    # Shared LMDB environments keyed by (pid, path). py-lmdb refuses to open the
    # same path twice in a single process (e.g. when both train_dataset and
    # test_dataset point at the same LMDB), so we cache Environment handles. The
    # PID in the key makes the cache fork-safe: worker processes spawned by the
    # dataloader see an empty cache for their own pid and open a fresh handle.
    _env_pool: dict = {}
    # Cached dir_cache per path so the LMDB only has to be opened once per
    # process for metadata even when multiple Dataset instances (train + test)
    # are built before the dataloader fork.
    _dir_cache_pool: dict = {}

    def __init__(
        self,
        data_root='/data/zhongyao/dataset',
        verbose=False,
        split=False,
        shift_range=20,
        **kwargs
    ):
        super().__init__(**kwargs)

        assert data_root is not None

        self.verbose = verbose
        self.dataset_label = 'googlestreet'
        mode = self.mode
        self.data_root = data_root
        # v2 LMDB layout: one pickle blob per view, pre-resized RGB/depth and
        # pre-scaled K. Produced by scripts/pack_googlestreet_lmdb.py.
        # Use the SSD copy under /home (nvme) instead of the HDD copy under /data.
        self.lmdb_path = '/home/wangqw/NeurIPS26/dataset_lmdb_v2'
        if not os.path.exists(self.lmdb_path):
            raise FileNotFoundError(
                f"[{self.dataset_label}] LMDB not found at {self.lmdb_path}. "
                "Run scripts/pack_googlestreet_lmdb.py first."
            )

        if self.verbose:
            print(f"[{self.dataset_label}] using LMDB at {self.lmdb_path}")

        # Load dir_cache eagerly so we can filter file_paths to only the folders
        # that actually made it into the LMDB. We cache per-path so that train
        # and test dataset instances constructed back-to-back don't both open
        # the LMDB (py-lmdb forbids re-opening the same env in one process
        # until the previous handle is fully GC'd, which is timing-dependent).
        cached = GoogleStreetDataset._dir_cache_pool.get(self.lmdb_path)
        if cached is None:
            cached = self._load_dir_cache_from_disk()
            GoogleStreetDataset._dir_cache_pool[self.lmdb_path] = cached
        self.dir_cache = cached

        all_paths = get_sorted_pair_paths(data_root, split=False, mode=mode)
        # 只使用 60% 的数据
        # all_paths = all_paths[:int(len(all_paths) * 0.6)]
        self.file_paths = [p for p in all_paths if p in self.dir_cache]
        if self.verbose:
            print(
                f"[{self.dataset_label}] {len(self.file_paths)}/{len(all_paths)} folders available in LMDB"
            )

        self.shift_range = shift_range
        self.sat_height = 5726
        self.sat_gap = 150

    def _load_dir_cache_from_disk(self) -> dict:
        """Open the LMDB briefly, read __VERSION__ + __DIR_CACHE__, close it."""
        import lmdb
        env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        try:
            with env.begin(buffers=True) as txn:
                version_bytes = txn.get(b'__VERSION__')
                if version_bytes is None:
                    raise RuntimeError(
                        f"[{self.dataset_label}] LMDB at {self.lmdb_path} is missing __VERSION__. "
                        "Repack with scripts/pack_googlestreet_lmdb.py."
                    )
                version = int(bytes(version_bytes).decode('utf-8'))
                if version != 2:
                    raise RuntimeError(
                        f"[{self.dataset_label}] unsupported LMDB version {version}; expected 2."
                    )
                dir_cache_bytes = txn.get(b'__DIR_CACHE__')
                if dir_cache_bytes is None:
                    return {}
                return json.loads(bytes(dir_cache_bytes).decode('utf-8'))
        finally:
            env.close()

    def _get_env(self):
        """Return the LMDB Environment for the current process, opening it on
        first use. Re-entrant across multiple dataset instances in the same
        process (they share the same handle)."""
        pid = os.getpid()
        key = (pid, self.lmdb_path)
        env = GoogleStreetDataset._env_pool.get(key)
        if env is None:
            import lmdb
            env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            GoogleStreetDataset._env_pool[key] = env
        return env

    def _load_view_blob(self, txn, folder_path: str, prefix: str):
        """Load one view's pickle blob. Returns a dict with keys K, c2w, rgb, depth, s."""
        key = f"{folder_path}|{prefix}".encode('utf-8')
        raw = txn.get(key)
        if raw is None:
            raise KeyError(f"[{self.dataset_label}] Missing LMDB view: {folder_path}|{prefix}")
        # pickle.loads accepts any bytes-like, including the memoryview LMDB hands
        # back when buffers=True, so we skip the extra bytes() memcpy.
        return pickle.loads(raw)

    def __len__(self):
        return len(self.file_paths)

    def natural_key(self, string: str):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', string)]

    def _get_views(self, index, resolution, rng):
        # 随机选择返回 2 张或 3 张图
        n_views_target = self.frame_num
        if index >= len(self.file_paths):
            raise IndexError(f"Index {index} out of range. Dataset has {len(self.file_paths)} samples.")
        folder_path = str(self.file_paths[index])

        env = self._get_env()
        txn = env.begin(buffers=True)
        folder_info = self.dir_cache.get(folder_path)
        if folder_info is None:
            raise KeyError(f"[{self.dataset_label}] {folder_path} not in LMDB dir_cache")
        all_prefixes = folder_info['p']
        ref_c2w_height = float(folder_info['ry'])

        shift_east = rng.uniform(-1, 1) * self.shift_range
        shift_south = rng.uniform(-1, 1) * self.shift_range
        current_sat_meters = rng.uniform(70, 210)

        # Partition prefixes by view type.
        sat_files = [p for p in all_prefixes if 'satellite' in p]
        ground_files = [p for p in all_prefixes if ('ground' in p and 'satellite' not in p and 'pano' not in p)]
        uav_files = [p for p in all_prefixes if 'uav' in p]

        total_available = len(sat_files) + len(ground_files) + len(uav_files)
        if total_available <= 0:
            raise ValueError(f"[{self.dataset_label}] No valid views found under {folder_path}")

        n_views = min(n_views_target, total_available)

        sat_files_sorted = sorted(sat_files, key=self.natural_key)
        ground_files_sorted = sorted(ground_files, key=self.natural_key)
        uav_files_sorted = sorted(uav_files, key=self.natural_key)

        # 使用 packer 预先记录的参考地面相机高度构造世界坐标系。
        ref_c2w = np.eye(4, dtype=np.float32)
        ref_c2w[1, 3] = ref_c2w_height
        ref_w2c = np.linalg.inv(ref_c2w)

        if n_views == total_available:
            sat_sel, ground_sel, uav_sel = sat_files_sorted, ground_files_sorted, uav_files_sorted
        else:
            feasible: list[tuple[int, int, int]] = []
            for k_sat in range(1, min(len(sat_files_sorted), n_views) + 1):
                for k_ground in range(0, min(len(ground_files_sorted), n_views - k_sat) + 1):
                    k_uav = n_views - k_sat - k_ground
                    if 0 <= k_uav <= len(uav_files_sorted):
                        feasible.append((k_sat, k_ground, k_uav))
            k_sat, k_ground, k_uav = feasible[rng.integers(len(feasible))]
            sat_sel = rng.choice(sat_files_sorted, size=k_sat, replace=False).tolist() if k_sat > 0 else []
            ground_sel = rng.choice(ground_files_sorted, size=k_ground, replace=False).tolist() if k_ground > 0 else []
            uav_sel = rng.choice(uav_files_sorted, size=k_uav, replace=False).tolist() if k_uav > 0 else []


        # Ordering rule:
        # - Satellite views must be placed at the very front.
        # - Ground/drone views are randomly mixed after satellites.
        sat_part = sorted(sat_sel, key=self.natural_key)
        grd_uav_part = list(ground_sel) + list(uav_sel)
        rng.shuffle(grd_uav_part)  # random order among ground/uav only
        sorted_prefixes = sat_part + grd_uav_part
        # Safety: enforce final length.
        sorted_prefixes = sorted_prefixes[:n_views]

        # 分开存储：卫星视图 vs 地面/无人机视图
        satellite_views = []
        ground_drone_views = []

        for prefix in sorted_prefixes:
            # One LMDB get() per view; unpickle yields fresh numpy arrays.
            blob = self._load_view_blob(txn, folder_path, prefix)
            K = blob['K'].copy()
            c2w = blob['c2w']
            rgb = blob['rgb']         # (H, W, 3) uint8, pre-resized
            depth = blob['depth']     # (H, W) float32, pre-resized + pre-clipped
            is_sat = bool(blob['s'])

            # 以参考相机(第一张地面图)为世界坐标系，计算相对的外参。
            c2w = ref_w2c @ c2w

            if is_sat:
                # Sat RGB is already at SAT_TARGET (1024). Apply the random
                # meter-based crop + resize the same way as before — only now
                # sat_W / sat_H are the pre-resized dimensions (1024), which
                # keeps the focal-length formula self-consistent.
                sat_H, sat_W = rgb.shape[0], rgb.shape[1]
                sat_meter_per_pixel = SAT_RES / sat_H
                sat_target_size = int(current_sat_meters / sat_meter_per_pixel)

                pixel_shift_x = int(shift_east / sat_meter_per_pixel)
                pixel_shift_y = int(shift_south / sat_meter_per_pixel)

                center_x = sat_W // 2 + pixel_shift_x
                center_y = sat_H // 2 + pixel_shift_y

                crop_left = int(center_x - sat_target_size // 2)
                crop_top = int(center_y - sat_target_size // 2)

                assert crop_left >= 0 and crop_top >= 0 \
                    and crop_left + sat_target_size <= sat_W \
                    and crop_top + sat_target_size <= sat_H, \
                    f"Crop box out of bounds: left={crop_left}, top={crop_top}, target_size={sat_target_size}"

                # RGB crop + LANCZOS upscale back to 1024x1024. cv2.INTER_LANCZOS4
                # is the same quality class as PIL.LANCZOS but ~3x faster.
                rgb_crop = np.ascontiguousarray(
                    rgb[crop_top:crop_top + sat_target_size, crop_left:crop_left + sat_target_size]
                )
                rgb = cv2.resize(rgb_crop, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)

                # Update intrinsics + shift the pose.
                K[0, 0] *= sat_W / sat_target_size
                K[1, 1] *= sat_H / sat_target_size
                c2w = c2w.copy()
                c2w[0, 3] += shift_south
                c2w[2, 3] += shift_east

                # Depth: crop the matching region and nearest-upsample to 1024.
                depth_H, depth_W = depth.shape
                scale_x = depth_W / sat_W
                scale_y = depth_H / sat_H
                d_crop_left = int(crop_left * scale_x)
                d_crop_top = int(crop_top * scale_y)
                d_crop_w = int(sat_target_size * scale_x)
                d_crop_h = int(sat_target_size * scale_y)
                depth_crop = depth[d_crop_top:d_crop_top + d_crop_h, d_crop_left:d_crop_left + d_crop_w]
                depth = cv2.resize(depth_crop, (1024, 1024), interpolation=cv2.INTER_NEAREST)

            # 数据增强与统一后处理
            rgb, depth, K = self._crop_resize_if_necessary(
                rgb,
                depth,
                K,
                resolution,
                rng=rng,
                info=folder_path,
                sat=is_sat,
            )

            # 确保卫星图的深度始终为非负（对应相机坐标系 z>=0），
            # 这样后续在 loss 中就不用再依赖「卫星图在第 0 个视角」去做特殊裁剪。
            # if is_sat:
            #     tmp_sat_height = -c2w[1,3]
            #     depth = np.clip(depth, a_min = tmp_sat_height - self.sat_gap, a_max = None)

            view_dict = dict(
                img=rgb,
                depthmap=depth.astype(np.float32),
                camera_pose=c2w.astype(np.float32),
                camera_intrinsics=K.astype(np.float32),
                sat_gap=self.sat_gap,
                dataset=self.dataset_label,
                label=f'mega_depth_{prefix}_{index}',
                instance=str(prefix + '_' + str(index)),
                sat_meters=current_sat_meters,
                sat_shift_east=shift_east if is_sat else 0,
                sat_shift_south=shift_south if is_sat else 0,
                is_satellite=bool(is_sat)
            )

            if is_sat:
                satellite_views.append(view_dict)
            else:
                ground_drone_views.append(view_dict)

        # 在各自列表内部打乱顺序
        if len(satellite_views) > 1:
            rng.shuffle(satellite_views)
        if len(ground_drone_views) > 1:
            rng.shuffle(ground_drone_views)


        # 可视化所有view在世界坐标系下的带颜色的点云，并保存为.ply文件
        # view_idx = [0, 1, 2]
        # self._save_colored_pointcloud_ply([views[i] for i in view_idx])

        # 返回带有分组信息的对象，后续在 BaseDataset 中再展开为扁平列表
        return {
            "satellite": satellite_views,      # 可以是 0 张、多张任意组合
            "ground_drone": ground_drone_views # 所有 ground / uav 视图
        }

    def _save_colored_pointcloud_ply(self, views):
        # 此部分与原代码完全保持一致，未做修改
        import os
        all_points = []
        all_colors = []

        for view_idx, view in enumerate(views):
            img = view['img']           
            if isinstance(img, Image.Image):
                img = np.array(img)
            depth = view['depthmap']    
            c2w = view['camera_pose']   
            K = view['camera_intrinsics'] 

            H, W = depth.shape
            valid_mask = depth > 0

            if not np.any(valid_mask):
                continue

            u, v = np.meshgrid(np.arange(W), np.arange(H))
            u = u[valid_mask]
            v = v[valid_mask]
            valid_depth = depth[valid_mask]
            valid_colors = img[valid_mask]  

            x_normalized = (u - K[0, 2]) / K[0, 0]
            y_normalized = (v - K[1, 2]) / K[1, 1]

            points_cam = np.stack([
                x_normalized * valid_depth,
                y_normalized * valid_depth,
                valid_depth
            ], axis=1)  

            points_cam_homo = np.concatenate([
                points_cam,
                np.ones((points_cam.shape[0], 1))
            ], axis=1)  

            points_world_homo = (c2w @ points_cam_homo.T).T  
            points_world = points_world_homo[:, :3]  

            all_points.append(points_world)
            all_colors.append(valid_colors)

        if len(all_points) == 0:
            return

        all_points = np.concatenate(all_points, axis=0)  
        all_colors = np.concatenate(all_colors, axis=0)  

        output_dir = 'data/vis_ply'
        os.makedirs(output_dir, exist_ok=True) # 加上这行确保文件夹存在
        output_path = os.path.join(output_dir, f'dataloader_pointcloud.ply')

        self._write_ply(output_path, all_points, all_colors)
        if self.verbose:
            print(f"Saved colored pointcloud to {output_path}, total points: {all_points.shape[0]}")

    def _write_ply(self, filename, points, colors):
        N = points.shape[0]
        with open(filename, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {N}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for i in range(N):
                f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]} ")
                f.write(f"{int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}\n")