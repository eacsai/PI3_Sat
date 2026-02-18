import os
import numpy as np
import cv2
import open3d as o3d
from tqdm import tqdm

def reconstruct_and_save(folder_path):
    # 1. 扫描所有的位姿配置文件
    npy_configs = [f for f in os.listdir(folder_path) if f.endswith('_rgb.npy')]
    npy_configs = [f for f in npy_configs if '_1' in f]
    all_pts, all_rgb, all_seg = [], [], []

    print(f"[*] 正在处理文件夹: {folder_path}")

    for npy_name in tqdm(npy_configs):
        # --- 数据读取核心逻辑 ---
        prefix = npy_name.replace('_rgb.npy', '') # 提取前缀，如 'ground_1'

        # A. 加载内参(K)和外参(c2w)
        meta = np.load(os.path.join(folder_path, npy_name), allow_pickle=True).item()
        K, c2w = meta['intrinsics'], meta['c2w']

        # B. 加载深度图 (NPY格式)
        depth = np.load(os.path.join(folder_path, f"{prefix}_depth.npy"))

        # C. 匹配并加载 RGB 图像 (处理卫星图和其他图不同的命名规则)
        rgb_file = f"{prefix}.png" if "satellite" in prefix else f"{prefix}_rgb.jpg"
        rgb = cv2.imread(os.path.join(folder_path, rgb_file))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # D. 类别识别与颜色分配
        if "satellite" in prefix:
            seg_color = [1.0, 1.0, 0.0] # 黄色
        elif "uav" in prefix:
            seg_color = [0.0, 0.0, 1.0] # 蓝色
        else: # ground
            seg_color = [1.0, 0.0, 0.0] # 红色

        # --- 数据预处理 ---
        if rgb.shape[:2] != depth.shape: # 确保像素对齐
            rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]))

        # ================= [新增] 可视化地面图 RGB 和 深度图 =================
        # 这里的判断条件排除卫星图，只保留地面图 (也可以加上 'uav' 如果你想看无人机)
        if "ground" in prefix and "satellite" not in prefix:
            # 1. 保存 RGB 图像 (OpenCV保存时需要转回 BGR)
            save_rgb_path = f"{prefix}_vis_rgb.jpg"
            cv2.imwrite(save_rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            # 2. 保存深度图的可视化结果
            # 深度图通常是 float32，需要归一化到 0-255 才能显示
            valid_mask_vis = depth > 0  # 假设0是无效值
            if valid_mask_vis.any():
                d_min = depth[valid_mask_vis].min()
                d_max = depth[valid_mask_vis].max()
                
                # 归一化: (d - min) / (max - min) -> [0, 1]
                depth_norm = (depth - d_min) / (d_max - d_min + 1e-6)
                depth_norm = np.clip(depth_norm, 0, 1) # 截断防止溢出
                
                # 转为 0-255 的 uint8
                depth_uint8 = (depth_norm * 255).astype(np.uint8)
                
                # 应用伪彩色 (JET 颜色映射: 蓝=近, 红=远)
                depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
                
                # 将无效深度区域设为黑色
                depth_colormap[~valid_mask_vis] = 0
                
                save_depth_path = f"{prefix}_vis_depth.jpg"
                cv2.imwrite(save_depth_path, depth_colormap)
        # ===================================================================


        # --- 反投影计算 ---
        mask = depth > 0.5
        v, u = np.where(mask)
        u, v, z = u[::4], v[::4], depth[mask][::4] # 4倍下采样提高性能

        # 相机坐标系点
        x_c = (u - K[0, 2]) * z / K[0, 0]
        y_c = (v - K[1, 2]) * z / K[1, 1]
        pts_cam = np.stack([x_c, y_c, z], axis=-1)

        # 变换到世界坐标系 (EDS: P_w = R*P_c + t)
        pts_world = (pts_cam @ c2w[:3, :3].T) + c2w[:3, 3]

        # 收集数据
        all_pts.append(pts_world)
        all_rgb.append(rgb[v, u] / 255.0)
        all_seg.append(np.tile(seg_color, (len(pts_world), 1)))

    # --- 结果保存 ---
    if all_pts:
        final_pts = np.vstack(all_pts[:2])

        # 保存 RGB 点云
        pcd_rgb = o3d.geometry.PointCloud()
        pcd_rgb.points = o3d.utility.Vector3dVector(final_pts)
        pcd_rgb.colors = o3d.utility.Vector3dVector(np.vstack(all_rgb[:2]))
        o3d.io.write_point_cloud(os.path.join('./data/vis_ply', "reconstruction_rgb.ply"), pcd_rgb)

        # 保存 三色分割点云
        pcd_seg = o3d.geometry.PointCloud()
        pcd_seg.points = o3d.utility.Vector3dVector(final_pts)
        pcd_seg.colors = o3d.utility.Vector3dVector(np.vstack(all_seg))
        o3d.io.write_point_cloud(os.path.join('./data/vis_ply', "reconstruction_seg.ply"), pcd_seg)

        print(f"[√] 两个点云文件已保存至: {folder_path}")

if __name__ == "__main__":
    # 执行单个 Pair 文件夹
    reconstruct_and_save("/data/zhongyao/dataset/0008_pair/0008_45_30/pair_16/")