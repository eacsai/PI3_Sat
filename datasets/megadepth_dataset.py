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

colmap_to_opencv = np.array([
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1]
], dtype=np.float32)

def load_depth_exr(path):
    exr = OpenEXR.InputFile(path)
    header = exr.header()
    w = int(header['dataWindow'].max.x + 1)
    h = int(header['dataWindow'].max.y + 1)
    depth = np.frombuffer(exr.channel('Y'), dtype=np.float32).reshape(h, w)
    depth_map  = depth.copy()
    depth_map[depth_map < 0] = 0
    exr.close()
    return depth_map

class MegaDepthDataset(BaseDataset):
    def __init__(
        self,
        data_root='/data/zhongyao/aer-grd-map/train_files_1024.txt',
        verbose=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        assert data_root is not None

        self.verbose = verbose
        self.dataset_label = 'MegaDepth'
        mode = self.mode
        self.data_root = data_root

        with open(self.data_root, 'r') as f:
            self.lines = f.readlines()
        # Remove empty lines and strip whitespace
        self.lines = [line.strip() for line in self.lines if line.strip()]

        # Parse file paths for all samples
        self.file_paths = []
        for line in self.lines:
            file_line = line.split(' ')
            assert len(file_line) >= 5
            grd_path, drone_path, sat_path, grd_mask, drone_mask = file_line[0], file_line[1], file_line[2], file_line[3], file_line[4]
            if mode == 'test':
                self.file_paths.append({
                    'ground': grd_path,
                    'drone': drone_path,
                    'satellite': sat_path,
                    'grd_mask': grd_mask,
                    'drone_mask': drone_mask,
                    'gt_shift_x': float(file_line[5]),
                    'gt_shift_y': float(file_line[6]),
                })
            else:
                self.file_paths.append({
                    'ground': grd_path,
                    'drone': drone_path,
                    'satellite': sat_path,
                    'grd_mask': grd_mask,
                    'drone_mask': drone_mask,
                    'gt_shift_x': np.random.uniform(-1, 1),
                    'gt_shift_y': np.random.uniform(-1, 1),
                })

        fx = 256.0  # focal length x
        fy = 256.0  # focal length y
        cx = 255.5  # optical center x
        cy = 255.5  # optical center y

        # width = 640
        # height = 480

        self.intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        self.num_imgs = 2 # per scene
        self.sat_height = 100.0 # meters


    def __len__(self):
        return len(self.lines)
                    
    def _get_views(self, index, resolution, rng):
        if index >= len(self.file_paths):
            raise IndexError(f"Index {index} out of range. Dataset has {len(self.file_paths)} samples.")

        paths = self.file_paths[index]
        views = []

        for key in ['ground', 'drone']:
            impath = paths[key]
            rgb_image = np.array(Image.open(impath)) # numpy shape (H, W, 3), uint8
            depth_path = impath.replace('.jpg', '.exr')
            depthmap = load_depth_exr(depth_path) # numpy shape (H, W)
            depthmap[depthmap > 300] = -1 # cap depth to 300 meters

            # load camera params
            npz_path = impath.replace('.jpeg.jpg', '.jpeg.npz')
            camera_pose = np.load(npz_path)['cam2world'].astype(np.float32)
            camera_pose = colmap_to_opencv @ camera_pose
            camera_intrinsics = np.load(npz_path)['intrinsics'].astype(np.float32)

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, camera_intrinsics, resolution, rng=rng, info=impath)


            # ==================== 可视化 RGB 和深度图 ====================
            # # 只可视化前几张图片避免过多输出
            # vis_dir = 'data/visualization'
            # os.makedirs(vis_dir, exist_ok=True)

            # import matplotlib.pyplot as plt
            # from matplotlib import cm

            # # 可视化 RGB 图像
            # fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # # RGB 图像
            # axes[0].imshow(rgb_image)
            # axes[0].set_title(f'RGB Image - {key}')
            # axes[0].axis('off')

            # # 深度图 (使用颜色映射)
            # depth_vis = depthmap.copy()
            # # 将无效深度值设为 NaN 以便显示为透明或特定颜色
            # depth_vis[depth_vis < 0] = np.nan
            # im = axes[1].imshow(depth_vis, cmap='turbo', vmin=0, vmax=80)
            # axes[1].set_title(f'Depth Map - {key}')
            # axes[1].axis('off')
            # plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

            # plt.tight_layout()
            # save_path = os.path.join(vis_dir, f'megadepth_{key}_index{index}.png')
            # plt.savefig(save_path, dpi=100, bbox_inches='tight')
            # plt.close()
            # ==================== 可视化结束 ====================

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset=self.dataset_label,
                label=f'mega_depth_{key}_{index}',
                instance=str(key + str(index)),
            ))
        
        return views

