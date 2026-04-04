import sys
sys.path.append('.')

from datasets.base.base_dataset import BaseDataset
import os
import numpy as np
import os.path as osp
from PIL import Image
import torchvision.transforms.functional as TF
from datasets.base.transforms import *
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
import random
from pathlib import Path
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
        exclude_suffixes = ('0013_pair')
    else:
        target_suffixes = ('0013_pair',)

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
        self.file_paths = get_sorted_pair_paths(data_root, split=False, mode=mode)
        self.shift_range = shift_range 
        self.sat_height = 5726
        self.sat_gap = 60

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
        npy_configs = [f for f in os.listdir(folder_path) if f.endswith('_rgb.npy')]

        shift_east = rng.uniform(-1, 1) * self.shift_range
        shift_south = rng.uniform(-1, 1) * self.shift_range
        current_sat_meters = rng.uniform(70, 210)

        # Collect all candidate views in this folder, then sample an arbitrary
        # satellite/ground/uav combination.
        sat_files = [f for f in npy_configs if 'satellite' in f]
        ground_files = [f for f in npy_configs if ('ground' in f and 'satellite' not in f and 'pano' not in f)]
        uav_files = [f for f in npy_configs if 'uav' in f]

        total_available = len(sat_files) + len(ground_files) + len(uav_files)
        if total_available <= 0:
            raise ValueError(f"[{self.dataset_label}] No valid *_rgb.npy views found under {folder_path}")

        n_views = min(n_views_target, total_available)

        sat_files_sorted = sorted(sat_files, key=self.natural_key)
        ground_files_sorted = sorted(ground_files, key=self.natural_key)
        uav_files_sorted = sorted(uav_files, key=self.natural_key)

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
        sorted_files = sat_part + grd_uav_part
        # Safety: enforce final length.
        sorted_files = sorted_files[:n_views]

        # 分开存储：卫星视图 vs 地面/无人机视图
        satellite_views = []
        ground_drone_views = []

        for idx, npy_name in enumerate(sorted_files):
            prefix = npy_name.replace('_rgb.npy', '')

            # A. 加载内参(K)和外参(c2w)
            meta = np.load(os.path.join(folder_path, npy_name), allow_pickle=True).item()
            K, c2w = meta['intrinsics'].copy(), meta['c2w'].copy()

            # B. 极速加载深度图
            if 'ground' in prefix and 'satellite' not in prefix:  
                # 优化点 1: weights_only=True 加速安全反序列化
                temp_depth_tensor = torch.load(
                    os.path.join(folder_path, f"{prefix}_depth_dap.pt"), 
                    map_location='cpu', 
                    weights_only=True
                )
                # 优化点 2: 用 .any() 极速判断张量是否非空
                if not temp_depth_tensor.any():
                    depth = np.load(os.path.join(folder_path, f"{prefix}_depth.npy")).astype(np.float32)
                else:
                    depth = temp_depth_tensor.detach().numpy().astype(np.float32)
                # depth = np.load(os.path.join(folder_path, f"{prefix}_depth.npy")).astype(np.float32)
                depth[depth > 60] = -1
            else:
                depth = np.load(os.path.join(folder_path, f"{prefix}_depth.npy")).astype(np.float32)
                if 'uav' in prefix:
                    depth[depth > 300] = -1
            # C. 安全加载并裁剪 RGB 图像
            rgb_file = f"{prefix}.jpg" if "satellite" in prefix else f"{prefix}_rgb.jpg"
            rgb_path = os.path.join(folder_path, rgb_file)
            
            # 优化点 3: 使用 with 语句，确保文件句柄在使用后立刻释放，防止系统 Cache 溢出
            with Image.open(rgb_path) as img:
                rgb = img.convert('RGB')
                
                if "satellite" in prefix:
                    is_sat = True
                    sat_H, sat_W = rgb.size[1], rgb.size[0]
                    sat_meter_per_pixel = SAT_RES / sat_H  
                    sat_target_size = int(current_sat_meters / sat_meter_per_pixel)

                    pixel_shift_x = int(shift_east / sat_meter_per_pixel)
                    pixel_shift_y = int(shift_south / sat_meter_per_pixel)
                    
                    center_x = sat_W // 2 + pixel_shift_x
                    center_y = sat_H // 2 + pixel_shift_y

                    crop_left = int(center_x - sat_target_size // 2)
                    crop_top = int(center_y - sat_target_size // 2)

                    assert crop_left >= 0 and crop_top >= 0 and crop_left + sat_target_size <= sat_W and crop_top + sat_target_size <= sat_H, \
                        f"Crop box out of bounds: left={crop_left}, top={crop_top}, target_size={sat_target_size}"
                    
                    # RGB 裁剪与缩放
                    rgb = TF.crop(rgb, crop_top, crop_left, sat_target_size, sat_target_size)
                    rgb = rgb.resize((1024, 1024), resample=Image.LANCZOS)                

                    # 更新相机内外参
                    K[0, 0] *= sat_W / sat_target_size  
                    K[1, 1] *= sat_H / sat_target_size  
                    c2w[0, 3] += shift_south
                    c2w[2, 3] += shift_east

                    # 优化点 4: 彻底摒弃 F.interpolate 产生巨大无用张量的逻辑
                    # 采用先按比例推算小图坐标 -> 小图上直接裁剪 -> cv2 高速放大的策略
                    depth_H, depth_W = depth.shape
                    scale_x = depth_W / sat_W
                    scale_y = depth_H / sat_H

                    d_crop_left = int(crop_left * scale_x)
                    d_crop_top = int(crop_top * scale_y)
                    d_crop_w = int(sat_target_size * scale_x)
                    d_crop_h = int(sat_target_size * scale_y)

                    depth_crop = depth[d_crop_top : d_crop_top + d_crop_h, d_crop_left : d_crop_left + d_crop_w]
                    
                    # 使用 cv2 极速最近邻插值到 1024x1024
                    depth = cv2.resize(depth_crop, (1024, 1024), interpolation=cv2.INTER_NEAREST)

                else:
                    is_sat = False

                # 将最终的 RGB 转回 numpy 数组
                rgb = np.array(rgb)

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