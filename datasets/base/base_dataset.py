from datasets.base.easy_dataset import EasyDataset
from pi3.utils.geometry import depthmap_to_absolute_camera_coordinates, homogenize_points, se3_inverse, satellite_depthmap_to_absolute_camera_coordinates
import numpy as np
import os
import PIL
import pi3.utils.cropping as cropping
import torchvision.transforms as tvf
from omegaconf import OmegaConf
from .transforms import *
import pandas as pd
from .utils import *

import plyfile
from plyfile import PlyData, PlyElement

class BaseDataset(EasyDataset):
    def __init__(
        self,
        seed=2024,
        resolution=None,            # (width, height) or list of (width, height) or list of int
        aug_crop=False,             # False or int, slightly scale the image a bit larger than the target resolution
        aug_focal=False,            # False or float in [0, 1]
        z_far=0,
        frame_num=2,
        transform=tvf.ToTensor(),
        cache_file=None,
        save_cache=False,
        mode='train',
        cache_name=None,
        max_refetch=3,
        random_sample_thres=0.1,
        shuffle=True,
        use_sparse_depth=False,
    ):
        super().__init__()
        self.frame_num = frame_num

        self.transform = transform

        self.shuffle = shuffle

        self.use_sparse_depth = use_sparse_depth

        self._rng_seed = int(seed)
        self._epoch = 0
        self._rng = np.random.default_rng(self._rng_seed)
        self._set_resolutions(resolution)

        self.aug_crop = aug_crop
        self.aug_focal = aug_focal

        self.z_far = z_far

        self.dataset_label = 'BaseDataset'

        self.save_cache = save_cache
        self.cache_loaded = False
        self.cache_name = cache_name
        if cache_file is not None:
            print(f'[BaseDataset] Loading cache from {cache_file}..')
            res = self.load_cache(cache_file)
            if res:
                self.cache_loaded = True
                print(f'[BaseDataset] Cache is loaded.')

        self.mode = mode
        self.max_refetch = max_refetch

        self.random_sample_thres = random_sample_thres  # default not to do that

    def set_epoch(self, epoch, base_seed=None):
        """每个 epoch 更新一次；与 pi3_trainer.before_epoch 中 dataset.set_epoch 对齐。"""
        if base_seed is not None:
            self._rng_seed = int(base_seed)
        self._epoch = int(epoch)
        # 全局 rng：随 epoch 变化（用于未改为 per-index 的路径）
        self._rng = np.random.default_rng(self._rng_seed + self._epoch * 7919)

    def _rng_for_index(self, idx):
        """同一 idx 在不同 epoch 使用不同 Generator；同一 epoch 内可复现。"""
        if isinstance(idx, tuple):
            idx0 = int(idx[0])
        else:
            idx0 = int(idx)
        # 大质数混合，避免 epoch/idx 低位模式重叠
        s = (self._rng_seed + self._epoch * 1000003 + idx0 * 9176) & 0xFFFFFFFFFFFFFFFF
        return np.random.default_rng(s)

    def convert_attributes(self):
        """
        Avoid memory leak caused by python list or python dict
        https://github.com/pytorch/pytorch/issues/13246
        """

        def _is_equivalent(original, converted):
            """
            Check if the converted data structure is equivalent to the original.
            """
            try:
                return original == converted
            except Exception:
                return False

        for attr_name in dir(self):
            if attr_name.startswith("__") or callable(getattr(self, attr_name)):
                continue
            
            attr_value = getattr(self, attr_name)
            
            if isinstance(attr_value, list):
                try:
                    converted_value = np.array(attr_value)
                    
                    if _is_equivalent(attr_value, converted_value.tolist()):
                        setattr(self, attr_name, converted_value)
                    else:
                        print(f"[{self.dataset_label}] <{attr_name}> conversion may not be equivalent, skipping.", flush=True)
                except ValueError as e:
                    print(f"[{self.dataset_label}] Error converting <{attr_name}>: {e}", flush=True)
            
            elif isinstance(attr_value, dict):
                try:
                    converted_value = pd.Series(attr_value)
                    if _is_equivalent(attr_value, converted_value.to_dict()):
                        setattr(self, attr_name, converted_value)
                    else:
                        print(f"[{self.dataset_label}] <{attr_name}> conversion may not be equivalent, skipping.", flush=True)
                except ValueError as e:
                    print(f"[{self.dataset_label}] Error converting <{attr_name}>: {e}", flush=True)

    def _set_resolutions(self, resolutions):
        assert resolutions is not None, 'undefined resolution'
        if OmegaConf.is_config(resolutions):
            resolutions = OmegaConf.to_object(resolutions)

        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            assert isinstance(width, int), f'Bad type for {width=} {type(width)=}, should be int'
            assert isinstance(height, int), f'Bad type for {height=} {type(height)=}, should be int'
            # assert width >= height
            # self._resolutions.append((width, height))
            self._resolutions.append([width, height])

        self.num_resoluions = len(self._resolutions)

    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, rng=None, info=None, normal=None, far_mask=None, sat=False):
        """ This function:
            - first downsizes the image with LANCZOS inteprolation,
              which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W-cx)
        min_margin_y = min(cy, H-cy)
        assert min_margin_x > W/5, f'Bad principal point in view={info}'
        assert min_margin_y > H/5, f'Bad principal point in view={info}'
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, depthmap, intrinsics, normal, far_mask = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, normal=normal)

        # transpose the resolution if necessary
        W, H = image.size  # new size
        # NOTE: Here we don't care about portrait image.
        # assert resolution[0] >= resolution[1]
        # if H > 1.1*W:
        #     # image is portrait mode
        #     resolution = resolution[::-1]
        # elif 0.9 < H/W < 1.1 and resolution[0] != resolution[1]:
        #     # image is square, so we chose (portrait, landscape) randomly
        #     if rng.integers(2):
        #         resolution = resolution[::-1]

        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if not sat:
            if self.aug_focal:
                crop_scale = self.aug_focal + (1.0 - self.aug_focal) * np.random.beta(0.5, 0.5) # beta distribution, bi-modal
                image, depthmap, intrinsics, normal, far_mask = cropping.center_crop_image_depthmap(image, depthmap, intrinsics, crop_scale, normal=normal, far_mask=far_mask)

            if self.aug_crop > 1:
                target_resolution += rng.integers(0, self.aug_crop)
        image, depthmap, intrinsics, normal, far_mask = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution, normal=normal, far_mask=far_mask) # slightly scale the image a bit larger than the target resolution

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, depthmap, intrinsics2, normal, far_mask = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, normal=normal, far_mask=far_mask)

        other = [x for x in [normal, far_mask] if x is not None]
        return image, depthmap, intrinsics2, *other
    
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            if len(idx) == 3:
                idx, ar_idx, frame_num = idx
                self.frame_num = frame_num
            else:
                idx, ar_idx = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0

        # over-loaded code
        resolution = self._resolutions[ar_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        
        error = None
        sample_rng = self._rng_for_index(idx)
        for _ in range(10):              # default: 3
            try:
                views = self._get_views(idx, resolution, sample_rng)

                # 兼容新格式：_get_views 可能返回 dict{'satellite': [...], 'ground_drone': [...]}
                assert isinstance(views, dict), f"views should be a dict, but got {type(views)}"
                sat_list = views.get("satellite", []) or []
                gd_list = views.get("ground_drone", []) or []
                views = sat_list + gd_list

                # assert len(views) == self.frame_num
                if self.shuffle:
                    sample_rng.shuffle(views)

                # check data-types
                for v, view in enumerate(views):
                    assert 'pts3d' not in view, f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
                    view['idx'] = (idx, ar_idx, v)
                    # view['idx'] = (idx, v)

                    # encode the image
                    width, height = view['img'].size
                    view['true_shape'] = np.int32((height, width))

                    assert 'camera_intrinsics' in view
                    if 'camera_pose' not in view:
                        view['camera_pose'] = np.full((4, 4), np.nan, dtype=np.float32)
                    else:
                        assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {view_name(view)}'
                    assert 'pts3d' not in view
                    assert 'valid_mask' not in view
                    assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for view {view_name(view)}'
                    view['z_far'] = self.z_far
                    if 'satellite' in view['label']:
                        view['camera_pose_ori'] = view['camera_pose'] # cam2world
                        camera_pose_new = view['camera_pose'].copy()
                        min_pose_y = min(item['camera_pose'][1, 3] for item in gd_list)
                        camera_pose_new[1,3] = min_pose_y - view['sat_gap']
                        view['camera_pose_new'] = se3_inverse(camera_pose_new) # world2cam
                        pts3d, valid_mask = satellite_depthmap_to_absolute_camera_coordinates(**view)
                        view['camera_pose'] = camera_pose_new # cam2world
                    else:
                        pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)

                    view['pts3d'] = pts3d
                    view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)

                    view['depthmap'][~view['valid_mask']] = 0.0

                    assert view['valid_mask'].sum() > 0

                    if 'normal' not in view:
                        view['normal'] = None

                    # # check all datatypes
                    # for key, val in view.items():
                    #     res, err_msg = is_good_type(key, val)
                    #     assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"
                    # K = view['camera_intrinsics']

                for view in views:
                    view['img'] = self.transform(view['img'])


                # # Visualize the point cloud
                # all_pts = []
                # all_colors = []
                
                # for i, view in enumerate(views):
                #     # pts3d is (H, W, 3)
                #     pts = view['pts3d']
                #     mask = view['valid_mask']
                    
                #     # Image is usually (3, H, W) after ToTensor transform and normalized [0,1]
                #     # We need to un-normalize to [0, 255] and reshape
                #     img_tensor = view['img']
                #     if img_tensor.ndim == 3: # C, H, W
                #         img_np = img_tensor.permute(1, 2, 0).numpy()
                #     else:
                #         img_np = np.array(img_tensor)
                        
                #     # Scale to 0-255 for ply colors
                #     if img_np.max() <= 1.0:
                #         img_np = (img_np * 255).astype(np.uint8)
                #     else:
                #         img_np = img_np.astype(np.uint8)

                #     # Flatten based on mask
                #     pts_masked = pts[mask]
                #     colors_masked = img_np[mask]

                #     # Apply camera pose if available to align point clouds in world space
                #     # If camera_pose is all NaN or Identity, this keeps it in camera space
                #     # transform to first frame camera coordinate
                #     pose = views[0].get('camera_pose', np.eye(4))
                #     w2c_target = se3_inverse(pose[None]) # (4, 4)
                #     w2c_target = torch.from_numpy(w2c_target).float()
                #     pts_masked = torch.einsum('bij, bnj -> bni', w2c_target, homogenize_points(torch.from_numpy(pts_masked[None]).float()))[..., :3].squeeze(0) # (N, 3)

                #     all_pts.append(pts_masked.numpy())
                #     all_colors.append(colors_masked)

                # # Concatenate all views
                # full_pts = np.concatenate(all_pts, axis=0)
                # full_colors = np.concatenate(all_colors, axis=0)

                # # Create structured array for PLY writing
                # vertex = np.zeros(full_pts.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                #                                             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
                
                # vertex['x'] = full_pts[:, 0]
                # vertex['y'] = full_pts[:, 1]
                # vertex['z'] = full_pts[:, 2]
                # vertex['red'] = full_colors[:, 0]
                # vertex['green'] = full_colors[:, 1]
                # vertex['blue'] = full_colors[:, 2]

                # # Save to current directory
                # ply_filename = "data/vis_ply/basedata_points.ply"
                # el = PlyElement.describe(vertex, 'vertex')
                # PlyData([el]).write(ply_filename)
                # print(f"[DEBUG] Saved point cloud to {os.path.abspath(ply_filename)} with {len(full_pts)} points")

            except Exception as e:
                views = None
                if hasattr(self, 'this_views_info'):
                    print(
                        f"Failed to load data from {self.dataset_label}-{idx} ({self.this_views_info}) for error {e}.", flush=True
                    )
                else:
                    print(
                        f"Failed to load data from {self.dataset_label}-{idx} for error {e}.", flush=True
                    )
                idx = np.random.randint(0, len(self))
                error = e
            
            if views is not None:
                error = None
                break

        if views is None:
            raise error
        
        return views
    
    def load_cache(self, cache_file):
        try:
            data = np.load(cache_file, allow_pickle=True).item()
            # Step 2: 遍历字典中的每个键值对，并将其赋值为类实例的属性
            if isinstance(data, dict):
                for key, value in data.items():
                    setattr(self, key, value)
                return True
            else:
                print("Error: The npy file does not contain a dictionary.")
                return False
        except Exception as e:
            print(f"An error occurred while loading the cache: {e}")
            return False

    def _save_cache(self, keys, desc=None):
        if desc is None:
            save_path = f'data/dataset_cache/{self.dataset_label}_cache.npy'
        else:
            save_path = f'data/dataset_cache/{self.dataset_label}_{desc}_cache.npy'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        save_dict = {}
        for key in keys:
            save_dict[key] = getattr(self, key)
        
        np.save(save_path, save_dict)

        print(f'Saved cache to {save_path}.', flush=True)

