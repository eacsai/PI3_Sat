import sys
sys.path.append('.')

from datasets.base.base_dataset import BaseDataset
import os
import numpy as np
import os.path as osp
from PIL import Image
from datasets.base.transforms import *
import json
from tqdm import tqdm
import OpenEXR
import torch.nn.functional as F
import random
from pathlib import Path
import re

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


def get_sorted_pair_paths(root_dir='.', split=True):
    """
    遍历指定目录结构，收集最底层的 pair_xxx 文件夹路径并排序。
    """
    target_paths = []

    # 1. 获取第一层目录：筛选以 "_pair" 结尾的文件夹
    # 例如: 0001_pair, 0005_pair
    if not os.path.exists(root_dir):
        print(f"错误: 目录 '{root_dir}' 不存在")
        return []

    level1_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    
    for l1_name in level1_names:
        if not l1_name.endswith('_pair'):
            continue
            
        l1_path = os.path.join(root_dir, l1_name)
        
        # 2. 获取第二层目录：直接读取里面的子文件夹
        # 例如: 0001_100_30, 0001_45_30
        level2_names = sorted([d for d in os.listdir(l1_path) if os.path.isdir(os.path.join(l1_path, d))])
        
        for l2_name in level2_names:
            l2_path = os.path.join(l1_path, l2_name)
            
            # 3. 获取第三层目录：筛选以 "pair_" 开头的文件夹
            # 例如: pair_0, pair_10
            level3_names = [d for d in os.listdir(l2_path) if os.path.isdir(os.path.join(l2_path, d))]
            
            for l3_name in level3_names:
                if l3_name.startswith('pair_'):
                    full_path = os.path.join(l2_path, l3_name)
                    target_paths.append(full_path)

    # 4. 自然排序逻辑 (Natural Sort)
    # 这一步是为了让 pair_2 排在 pair_10 前面，而不是后面
    def natural_key(string):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', string)]

    target_paths.sort(key=natural_key)
    
    if split:
        # 1. 生成带 _1 的列表
        paths_1 = [os.path.join(p, '_1') for p in target_paths]
        # 2. 生成带 _2 的列表
        paths_2 = [os.path.join(p, '_2') for p in target_paths]
        # 3. 合并两个列表
        final_paths = paths_1 + paths_2
    else:
        final_paths = target_paths

    return final_paths

class GoogleStreetDataset(BaseDataset):
    def __init__(
        self,
        data_root='/data/zhongyao/dataset',
        verbose=False,
        split=True,
        shift_range=20,
        **kwargs
    ):
        super().__init__(**kwargs)

        assert data_root is not None

        self.verbose = verbose
        self.dataset_label = 'googlestreet'
        mode = self.mode
        self.split = split
        self.file_paths = get_sorted_pair_paths(data_root, split=self.split)
        # self.file_paths.sort(key=lambda x: int(x.name.split('_')[-1]))
        self.shift_range = shift_range # 卫星图随机平移范围(米)
        self.sat_height = 5726
        self.sat_gap = 150

    def __len__(self):
        return len(self.file_paths)
                    
    def _get_views(self, index, resolution, rng):
        if index >= len(self.file_paths):
            raise IndexError(f"Index {index} out of range. Dataset has {len(self.file_paths)} samples.")

        if self.split:
            parts = self.file_paths[index].split('/')
            parts.insert(0, '/')
            folder_path = os.path.join(*parts[:-1])
            part = parts[-1]
        else:
            folder_path = str(self.file_paths[index])
            part = '_1'

        shift_east = np.random.uniform(-1, 1) * self.shift_range
        shift_south = np.random.uniform(-1, 1) * self.shift_range

        current_sat_meters = rng.uniform(70, 210)

        npy_configs = [f for f in os.listdir(folder_path) if f.endswith('_rgb.npy')]
        npy_configs = [f for f in npy_configs if part in f]
        sorted_files = sorted(npy_configs, key=get_sort_priority)

        views = []
        for idx, npy_name in enumerate(sorted_files):
            # --- 数据读取核心逻辑 ---
            prefix = npy_name.replace('_rgb.npy', '') # 提取前缀，如 'ground_1'

            # A. 加载内参(K)和外参(c2w)
            meta = np.load(os.path.join(folder_path, npy_name), allow_pickle=True).item()
            K, c2w = meta['intrinsics'].copy(), meta['c2w'].copy()

            # B. 加载深度图 (NPY格式)
            if 'ground' in prefix and 'satellite' not in prefix:  # 只有地面图有 DAP 格式的深度图
                # 1. 先读取 Tensor 到临时变量
                temp_depth_tensor = torch.load(
                    os.path.join(folder_path, f"{prefix}_depth_dap.pt"), 
                    map_location='cpu', 
                    weights_only=False
                )
                # 2. 检查是否全为 0 (有效性检查)
                # 使用 .max() == 0 或者 .sum() == 0 都可以判定全零
                if temp_depth_tensor.max() == 0:
                    # Case A: 如果全为 0，回退加载 .npy 文件
                    # print(f"Warning: {prefix}_depth_dap.pt is all zeros, falling back to .npy") # 可选：打印日志
                    depth = np.load(os.path.join(folder_path, f"{prefix}_depth.npy")).astype(np.float32)
                else:
                    # Case B: 如果数据有效，转为 numpy 并进行后处理
                    depth = temp_depth_tensor.detach().numpy().astype(np.float32)
                    depth[depth > 60] = -1
            else:
                depth = np.load(os.path.join(folder_path, f"{prefix}_depth.npy"))
            # depth = np.load(os.path.join(folder_path, f"{prefix}_depth.npy"))
            # C. 匹配并加载 RGB 图像 (处理卫星图和其他图不同的命名规则)
            rgb_file = f"{prefix}.png" if "satellite" in prefix else f"{prefix}_rgb.jpg"
            rgb = Image.open(os.path.join(folder_path, rgb_file)).convert('RGB')
            if "satellite" in prefix:
                sat_H, sat_W = rgb.size[1], rgb.size[0]
                sat_meter_per_pixel = SAT_RES / sat_H  # 假设卫星图是正方形
                sat_target_size = int(current_sat_meters / sat_meter_per_pixel) # 根据当前卫星图覆盖的实际范围计算目标裁剪尺寸

                # 假设图片坐标系: 右为东(+u), 下为南(+v)
                pixel_shift_x = int(shift_east / sat_meter_per_pixel)   # 东移
                pixel_shift_y = int(shift_south / sat_meter_per_pixel)  # 南移
                # 计算新的中心点坐标
                center_x = sat_W // 2 + pixel_shift_x
                center_y = sat_H // 2 + pixel_shift_y

                # 计算裁剪框左上角 (Left, Top)
                crop_left = int(center_x - sat_target_size // 2)
                crop_top = int(center_y - sat_target_size // 2)

                assert crop_left >= 0 and crop_top >= 0 and crop_left + sat_target_size <= sat_W and crop_top + sat_target_size <= sat_H, \
                    f"Crop box out of bounds: left={crop_left}, top={crop_top}, target_size={sat_target_size}, image_size=({sat_W}, {sat_H})"
                
                rgb = TF.crop(rgb, crop_top, crop_left, sat_target_size, sat_target_size)
                rgb = rgb.resize((1024, 1024), resample=Image.LANCZOS)                

                # 更新内参矩阵 K
                K[0, 0] *= sat_W / sat_target_size  # fx
                K[1, 1] *= sat_H / sat_target_size  # fy

                # --- 更新外参 c2w ---
                # 坐标系定义: X轴->正南, Y轴->垂直向下, Z轴->正东
                # 相机位置 = c2w[:3, 3] (平移向量)
                # 向正南移动 shift_south 米 -> X轴增加
                c2w[0, 3] += shift_south
                # 向正东移动 shift_east 米 -> Z轴增加
                c2w[2, 3] += shift_east

                # 同步更新深度
                ## 1. 先将深度恢复到原始分辨率
                depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
                depth_full = F.interpolate(depth_tensor, size=(sat_H, sat_W), mode='nearest').squeeze().numpy()
                # 2. 使用与 RGB 相同的坐标进行裁剪
                depth_crop = depth_full[crop_top : crop_top + sat_target_size, 
                                        crop_left : crop_left + sat_target_size]
                
                # 3. 缩放到 1024x1024
                depth_crop_tensor = torch.from_numpy(depth_crop).unsqueeze(0).unsqueeze(0)
                depth = F.interpolate(depth_crop_tensor, size=(1024, 1024), mode='nearest').squeeze().numpy()

            rgb = np.array(rgb)

            rgb, depth, K = self._crop_resize_if_necessary(
                rgb, depth, K, resolution, rng=rng, info=folder_path)

            views.append(dict(
                img=rgb,
                depthmap=depth.astype(np.float32),
                camera_pose=c2w.astype(np.float32),
                camera_intrinsics=K.astype(np.float32),
                dataset=self.dataset_label,
                label=f'mega_depth_{prefix}_{index}',
                instance=str(prefix + '_' + str(index)),
                sat_height=self.sat_height,
                sat_gap=self.sat_gap,
                sat_meters=current_sat_meters,
                sat_shift_east=shift_east if "satellite" in prefix else 0,
                sat_shift_south=shift_south if "satellite" in prefix else 0
            ))

        # 可视化所有view在世界坐标系下的带颜色的点云，并保存为.ply文件
        # view_idx = [0, 1, 2]
        # self._save_colored_pointcloud_ply([views[i] for i in view_idx])

        lst = [1, 2]
        random.shuffle(lst)
        lst.insert(0, 0)  # 确保卫星图始终在第一位
        views = [views[i] for i in lst]  # 按照卫星图、地面图、无人机图或卫星图、无人机、地面图的顺序返回
        return views

    def _save_colored_pointcloud_ply(self, views):
        """将所有view在世界坐标系下的带颜色点云保存为.ply文件

        Args:
            views: 视图列表，每个view包含img, depthmap, camera_pose, camera_intrinsics
        """
        import os

        all_points = []
        all_colors = []

        for view_idx, view in enumerate(views):
            img = view['img']           # (H, W, 3) RGB图像
            # 确保img是numpy数组
            if isinstance(img, Image.Image):
                img = np.array(img)
            depth = view['depthmap']    # (H, W) 深度图
            c2w = view['camera_pose']   # (4, 4) 相机到世界坐标系的变换矩阵
            K = view['camera_intrinsics']  # (3, 3) 内参矩阵

            H, W = depth.shape

            # 获取有效的深度点
            valid_mask = depth > 0

            if not np.any(valid_mask):
                continue

            # 像素坐标 (u, v)
            u, v = np.meshgrid(np.arange(W), np.arange(H))
            u = u[valid_mask]
            v = v[valid_mask]
            valid_depth = depth[valid_mask]
            valid_colors = img[valid_mask]  # (N, 3) RGB颜色

            # 归一化相机坐标系 (x, y, 1)
            x_normalized = (u - K[0, 2]) / K[0, 0]
            y_normalized = (v - K[1, 2]) / K[1, 1]

            # 相机坐标系下的3D点
            points_cam = np.stack([
                x_normalized * valid_depth,
                y_normalized * valid_depth,
                valid_depth
            ], axis=1)  # (N, 3)

            # 添加齐次坐标
            points_cam_homo = np.concatenate([
                points_cam,
                np.ones((points_cam.shape[0], 1))
            ], axis=1)  # (N, 4)

            # 转换到世界坐标系
            points_world_homo = (c2w @ points_cam_homo.T).T  # (N, 4)
            points_world = points_world_homo[:, :3]  # (N, 3)

            all_points.append(points_world)
            all_colors.append(valid_colors)

        if len(all_points) == 0:
            return

        # 合并所有点
        all_points = np.concatenate(all_points, axis=0)  # (M, 3)
        all_colors = np.concatenate(all_colors, axis=0)  # (M, 3)

        # 保存为.ply文件
        output_dir = 'data/vis_ply'
        output_path = os.path.join(output_dir, f'dataloader_pointcloud.ply')

        self._write_ply(output_path, all_points, all_colors)
        if self.verbose:
            print(f"Saved colored pointcloud to {output_path}, total points: {all_points.shape[0]}")

    def _write_ply(self, filename, points, colors):
        """写入PLY格式的点云文件

        Args:
            filename: 输出文件路径
            points: (N, 3) 点坐标数组
            colors: (N, 3) RGB颜色数组，值范围0-255
        """
        N = points.shape[0]

        with open(filename, 'w') as f:
            # PLY文件头
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {N}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            # 写入点云数据
            for i in range(N):
                f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]} ")
                f.write(f"{int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}\n")

