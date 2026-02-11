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

class GoogleStreetDataset(BaseDataset):
    def __init__(
        self,
        data_root='/data/zhongyao/dataset/0005_pair/0005_70_30',
        verbose=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        assert data_root is not None

        self.verbose = verbose
        self.dataset_label = 'googlestreet'
        mode = self.mode
        self.data_root = Path(data_root)

        self.file_paths = [p for p in self.data_root.iterdir() if p.is_dir()]
        self.file_paths.sort(key=lambda x: int(x.name.split('_')[-1]))
        self.sat_height = 5726
        self.sat_gap = 150

    def __len__(self):
        return len(self.file_paths)
                    
    def _get_views(self, index, resolution, rng):
        if index >= len(self.file_paths):
            raise IndexError(f"Index {index} out of range. Dataset has {len(self.file_paths)} samples.")

        folder_path = str(self.file_paths[index])
        gt_shift_x = np.random.uniform(-1, 1)
        gt_shift_y = np.random.uniform(-1, 1)

        npy_configs = [f for f in os.listdir(folder_path) if f.endswith('_rgb.npy')]
        npy_configs = [f for f in npy_configs if '_1' in f]
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
                depth = torch.load(os.path.join(folder_path, f"{prefix}_depth_dap.pt"), weights_only=False).cpu().detach().numpy().astype(np.float32)
                depth[depth > 60] = -1
            else:
                depth = np.load(os.path.join(folder_path, f"{prefix}_depth.npy"))

            # C. 匹配并加载 RGB 图像 (处理卫星图和其他图不同的命名规则)
            rgb_file = f"{prefix}.png" if "satellite" in prefix else f"{prefix}_rgb.jpg"
            rgb = Image.open(os.path.join(folder_path, rgb_file)).convert('RGB')
            if "satellite" in prefix:
                sat_H, sat_W = rgb.size[1], rgb.size[0]
                sat_meter_per_pixel = SAT_RES / sat_H  # 假设卫星图是正方形
                sat_target_size = int(SAT_METERS / sat_meter_per_pixel)
                rgb = TF.center_crop(rgb, (sat_target_size, sat_target_size))
                rgb = rgb.resize((1024, 1024), resample=lanczos)
                # 更新内参矩阵 K
                K[0, 0] *= sat_W / sat_target_size  # fx
                K[1, 1] *= sat_H / sat_target_size  # fy
                # 同步更新深度
                ## 1. 先将深度恢复到原始分辨率
                depth = F.interpolate(torch.from_numpy(depth).unsqueeze(0).unsqueeze(0), size=(sat_H, sat_W), mode='nearest').squeeze().numpy()
                ## 2. 再进行中心裁剪和缩放
                depth = depth[(sat_H - sat_target_size) // 2 : (sat_H + sat_target_size) // 2, (sat_W - sat_target_size) // 2 : (sat_W + sat_target_size) // 2]
                depth = F.interpolate(torch.from_numpy(depth).unsqueeze(0).unsqueeze(0), size=(1024, 1024), mode='nearest').squeeze().numpy()

            rgb = np.array(rgb)

            rgb, depth, K = self._crop_resize_if_necessary(
                rgb, depth, K, resolution, rng=rng, info=folder_path)

            views.append(dict(
                img=rgb,
                depthmap=depth,
                camera_pose=c2w,
                camera_intrinsics=K.astype(np.float32),
                dataset=self.dataset_label,
                label=f'mega_depth_{prefix}_{index}',
                instance=str(prefix + '_' + str(index)),
                sat_height=self.sat_height,
                sat_gap=self.sat_gap
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

