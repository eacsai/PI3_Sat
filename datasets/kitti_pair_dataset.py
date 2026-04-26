"""KITTI cross-view (sat + ground) dataset for Pi3 fine-tuning.

This is a *training-side* adapter that reads the same KITTI list files as
``Pi3_eval/Pi3/datasets/kitti_pair.py`` but plugs into Pi3's BaseDataset
pipeline (with the sat_gap / two-camera convention used by googlestreet).

Key differences vs the eval-side ``kitti_pair.py``:
  * Inherits ``BaseDataset`` so it works with the Pi3 trainer / collator.
  * No GT depth → returns dummy ``depthmap`` filled with a benign positive
    value so the BaseDataset assertions (``valid_mask.sum() > 0``) pass.
    The KITTI loss (``pi3.models.loss_kitti.Pi3KittiLoss``) ignores all
    point-cloud terms and uses only camera-pose + sat-projection
    consistency, so the dummy depth never enters any real gradient path.
  * Camera-pose convention matches the **googlestreet** dataset (so the
    C1_highres weights interpret the input correctly without a coord
    re-mapping):
        - World axes: X=right, Y=down, Z=forward (vehicle heading at θ=0)
        - Sat camera at world (0, -150, 0) looking straight down with
          R_sat_c2w = [[0,1,0],[0,0,1],[1,0,0]]  (matches real LMDB sample).
        - Ground camera at (Y_g, 0, X_g) where (X_g, Y_g) come from the
          kitti_pair shift+theta math, remapped (kp_X→gs_Z, kp_Y→gs_X).
        - Ground rotation = R_y(θ) so heading at θ=0 = +Z forward.
        - sat_gap = 150 → BaseDataset overrides sat c2w[1,3] to -150.

Returns the same dict structure as ``GoogleStreetDataset._get_views``:
    {"satellite": [sat_view], "ground_drone": [grd_view]}
where each view dict carries ``img`` (np.uint8 HxWx3), ``depthmap``,
``camera_pose``, ``camera_intrinsics``, ``sat_gap``, ``is_satellite``,
``dataset``, ``label``, ``instance``, plus extra GT fields used by the
KITTI loss:
    ``kitti_sat_mpp``           — meter-per-pixel of the un-cropped sat
    ``kitti_ground_uv_in_sat``  — GT (u, v) of ground camera in sat image
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

from datasets.base.base_dataset import BaseDataset

# Reuse the eval-side ``data_utils`` for KITTI constants (CameraGPS_shift_left,
# meter_per_pixel, satmap_sidelength, …).  This is a read-only import.
_EVAL_DATASETS_DIR = Path(
    "/home/wangqw/video_program/Pi3_eval/Pi3/datasets"
)
if str(_EVAL_DATASETS_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DATASETS_DIR))
import data_utils as kitti_utils  # noqa: E402

_DEFAULT_ROOT = "/data/dataset/KITTI_dataset/KITTI"
_DEFAULT_TRAIN_LIST = str(_EVAL_DATASETS_DIR / "train_files.txt")
_DEFAULT_TEST1_LIST = str(_EVAL_DATASETS_DIR / "test1_files.txt")
_DEFAULT_TEST2_LIST = str(_EVAL_DATASETS_DIR / "test2_files.txt")

_SATMAP_DIR = "satmap"
_GRDIMG_DIR = "depth_data"
_LEFT_COLOR_NO_SKY = "image_02/grd_no_sky"
_LEFT_COLOR_ORIG = "image_02/data"
_OXTS_DIR = "oxts/data"


class KittiPairDataset(BaseDataset):
    """KITTI cross-view sat+ground pair dataset for Pi3 fine-tuning.

    Args (BaseDataset args also accepted via **kwargs):
        root: KITTI dataset root.
        split: 'train' / 'test1' / 'test2'.  Picks default list file.
        list_file: explicit list path (overrides split-based lookup).
        shift_range_lat / shift_range_lon: lateral / longitudinal shift
            magnitude in meters (matches kitti_pair eval defaults: 20).
        rotation_range: max sat rotation in degrees (default 10).
        sat_height: synthetic sat altitude above ground in meters
            (default 150, matches user-specified setup; written into
            ``sat_gap`` so BaseDataset overrides sat c2w y to -150).
        camera_height: ground camera y in world (default 0; could be 1.65
            but BaseDataset doesn't surgically rewrite ground y).
        ground_isotropic_resize: same semantics as eval kitti_pair.
    """

    def __init__(
        self,
        root: str = _DEFAULT_ROOT,
        split: str = "train",
        list_file: str | None = None,
        shift_range_lat: float = 20.0,
        shift_range_lon: float = 20.0,
        rotation_range: float = 10.0,
        sat_height: float = 150.0,
        camera_height: float = 0.0,
        use_orig_ground: bool = False,
        ground_isotropic_resize: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset_label = "kitti_pair"
        self.root = root
        self.shift_range_lat = float(shift_range_lat)
        self.shift_range_lon = float(shift_range_lon)
        self.rotation_range = float(rotation_range)
        self.sat_height = float(sat_height)
        self.camera_height = float(camera_height)
        self.use_orig_ground = bool(use_orig_ground)
        self.ground_isotropic_resize = bool(ground_isotropic_resize)

        # KITTI sat scale + reference shift constants
        self.meter_per_pixel = float(kitti_utils.get_meter_per_pixel(scale=1))
        self.shift_pixels_lat = self.shift_range_lat / self.meter_per_pixel
        self.shift_pixels_lon = self.shift_range_lon / self.meter_per_pixel
        self.satmap_side = int(kitti_utils.get_process_satmap_sidelength())  # 512

        split = split.lower()
        if split not in ("train", "test1", "test2"):
            raise ValueError(
                f"split must be 'train' | 'test1' | 'test2', got {split!r}"
            )
        self.split = split
        if list_file is not None:
            list_path = list_file
        else:
            default = {
                "train": _DEFAULT_TRAIN_LIST,
                "test1": _DEFAULT_TEST1_LIST,
                "test2": _DEFAULT_TEST2_LIST,
            }[split]
            list_path = default
        if not os.path.exists(list_path):
            raise FileNotFoundError(
                f"[{self.dataset_label}] list file not found: {list_path}"
            )
        with open(list_path, "r") as f:
            self.lines = [ln.strip() for ln in f if ln.strip()]

    def __len__(self):
        return len(self.lines)

    # --- pose construction (matches googlestreet convention) ---------------
    def _build_sat_c2w(self) -> np.ndarray:
        """Sat camera at (0, -h_sat, 0) looking down. R matches real LMDB."""
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = np.array(
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32
        )
        c2w[:3, 3] = np.array([0.0, -self.sat_height, 0.0], dtype=np.float32)
        return c2w

    def _build_grd_c2w(self, gt_shift_x: float, gt_shift_y: float,
                       theta_deg: float) -> np.ndarray:
        """Ground camera position + heading in gs world axes."""
        # Same translation math as eval-side kitti_pair:
        X_pre = -gt_shift_x * self.shift_range_lon  # along heading (forward)
        Y_pre = -gt_shift_y * self.shift_range_lat  # perpendicular ("left")
        theta_rad = float(np.deg2rad(theta_deg))
        ct, st = float(np.cos(theta_rad)), float(np.sin(theta_rad))
        X_g_kp = ct * X_pre - st * Y_pre  # post-rotation in heading frame
        Y_g_kp = st * X_pre + ct * Y_pre

        # Map (kitti_pair X=heading-axis, Y=left, Z=up) → (gs X=right, Y=down, Z=forward)
        # by sending kp_X → gs_Z, kp_Y → gs_X (sign chosen so heading at θ=0 = +Z).
        gs_X = Y_g_kp
        gs_Y = self.camera_height  # 0 by default (ground at world origin in y)
        gs_Z = X_g_kp

        # Ground rotation: heading at θ=0 = +Z, after rotation = R_y(θ) applied.
        # forward in gs world = (sin θ, 0, cos θ)
        # down = (0, 1, 0)
        # right = down × forward = (cos θ, 0, -sin θ)
        forward = np.array([st, 0.0, ct], dtype=np.float32)
        down = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(down, forward).astype(np.float32)
        R_grd_c2w = np.stack([right, down, forward], axis=1)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R_grd_c2w
        c2w[:3, 3] = np.array([gs_X, gs_Y, gs_Z], dtype=np.float32)
        return c2w

    # --- main view loader --------------------------------------------------
    def _get_views(self, index, resolution, rng):
        out_w, out_h = int(resolution[0]), int(resolution[1])
        line = self.lines[index]
        parts = line.split(" ")
        file_name = parts[0]
        if len(parts) >= 4:
            gt_shift_x = -float(parts[1])
            gt_shift_y = -float(parts[2])
            theta_norm = float(parts[3])
        else:
            gt_shift_x = float(rng.uniform(-1, 1))
            gt_shift_y = float(rng.uniform(-1, 1))
            theta_norm = float(rng.uniform(-1, 1))
        theta_deg = theta_norm * self.rotation_range

        day_dir = file_name[:10]
        drive_dir = file_name[:38]
        image_no = file_name[38:]

        # 1) load sat
        sat_path = os.path.join(self.root, _SATMAP_DIR, file_name)
        with Image.open(sat_path, "r") as SatMap:
            sat_pil = SatMap.convert("RGB")

        # 2) heading from oxts
        oxts_path = os.path.join(
            self.root, _GRDIMG_DIR, drive_dir, _OXTS_DIR,
            image_no.lower().replace(".png", ".txt"),
        )
        with open(oxts_path, "r") as f:
            content = f.readline().split(" ")
        heading_rad = float(content[5])

        # 3) load ground (prefer no-sky variant)
        grd_no_sky = os.path.join(
            self.root, _GRDIMG_DIR, drive_dir, _LEFT_COLOR_NO_SKY,
            image_no.lower(),
        )
        grd_orig = os.path.join(
            self.root, _GRDIMG_DIR, drive_dir, _LEFT_COLOR_ORIG,
            image_no.lower(),
        )
        if self.use_orig_ground or not os.path.exists(grd_no_sky):
            grd_path = grd_orig
        else:
            grd_path = grd_no_sky
        if not os.path.exists(grd_path):
            raise FileNotFoundError(grd_path)
        with Image.open(grd_path, "r") as g:
            grd_pil = g.convert("RGB")

        # 4) sat preprocessing — heading-align, shift, rotate, crop
        sat_rot = sat_pil.rotate(-heading_rad / np.pi * 180)
        sat_align = sat_rot.transform(
            sat_rot.size, Image.AFFINE,
            (1, 0, kitti_utils.CameraGPS_shift_left[0] / self.meter_per_pixel,
             0, 1, kitti_utils.CameraGPS_shift_left[1] / self.meter_per_pixel),
            resample=Image.BILINEAR,
        )
        sat_shift = sat_align.transform(
            sat_align.size, Image.AFFINE,
            (1, 0, gt_shift_x * self.shift_pixels_lon,
             0, 1, -gt_shift_y * self.shift_pixels_lat),
            resample=Image.BILINEAR,
        )
        sat_rot2 = sat_shift.rotate(theta_deg)
        sat_512 = TF.center_crop(sat_rot2, self.satmap_side)
        # Center-crop (out_h, out_w) directly out of the 512×512 sat — keeps
        # pixels isotropic (so sat_mpp_x == sat_mpp_y == self.meter_per_pixel)
        # which is essential for non-square targets like 252×504.
        sat_final = TF.center_crop(sat_512, [out_h, out_w])
        sat_arr = np.asarray(sat_final, dtype=np.uint8)  # HxWx3

        # 5) ground preprocessing
        gw, gh = grd_pil.size
        if self.ground_isotropic_resize:
            scale = float(out_h) / float(gh)
            new_w = int(round(gw * scale))
            new_h = int(out_h)
            grd_resized = grd_pil.resize((new_w, new_h), Image.BILINEAR)
            if new_w >= out_w:
                grd_cropped = TF.center_crop(grd_resized, [out_h, out_w])
            else:
                canvas = Image.new("RGB", (out_w, out_h))
                canvas.paste(grd_resized, ((out_w - new_w) // 2, 0))
                grd_cropped = canvas
        else:
            # Anisotropic resize then center-crop, mirroring eval-side
            grd_resize_w = out_w * 4  # 4:1 aspect like KITTI's 1024x256
            grd_resize_h = out_h
            grd_resized = grd_pil.resize(
                (grd_resize_w, grd_resize_h), Image.BILINEAR,
            )
            grd_cropped = TF.center_crop(grd_resized, [out_h, out_w])
        grd_arr = np.asarray(grd_cropped, dtype=np.uint8)

        # 6) intrinsics
        # Sat: orthographic — fake pinhole with focal = h_sat / mpp.
        # We center-crop (no resize) so mpp stays at the original
        # KITTI sat scale regardless of (out_h, out_w).
        sat_mpp_final = self.meter_per_pixel
        K_sat = np.array(
            [[self.sat_height / sat_mpp_final, 0, out_w / 2.0],
             [0, self.sat_height / sat_mpp_final, out_h / 2.0],
             [0, 0, 1]], dtype=np.float32,
        )
        # Ground: read calib_cam_to_cam.txt for left color cam.
        calib_path = os.path.join(
            self.root, _GRDIMG_DIR, day_dir, "calib_cam_to_cam.txt",
        )
        fx0 = fy0 = cx0 = cy0 = None
        with open(calib_path, "r") as f:
            for ln in f:
                if "P_rect_02" in ln:
                    vals = ln.split(":")[1].strip().split(" ")
                    fx0 = float(vals[0])
                    cx0 = float(vals[2])
                    fy0 = float(vals[5])
                    cy0 = float(vals[6])
                    break
        if fx0 is None:
            raise RuntimeError(f"No P_rect_02 in {calib_path}")
        if self.ground_isotropic_resize:
            scale = float(out_h) / float(gh)
            fx = fx0 * scale
            fy = fy0 * scale
            new_w = int(round(gw * scale))
            cx = cx0 * scale - max(0, (new_w - out_w) // 2)
            cy = cy0 * scale
        else:
            sx = (out_w * 4) / gw
            sy = out_h / gh
            fx = fx0 * sx
            cx = cx0 * sx - (out_w * 4 - out_w) / 2.0
            fy = fy0 * sy
            cy = cy0 * sy
        # Force principal point inside image so BaseDataset's crop assertion
        # ``min_margin > W/5`` passes.  KITTI's principal point is centered
        # enough that this normally holds, but anisotropic squashing can push
        # it; clamp defensively.
        cx = float(np.clip(cx, out_w * 0.25, out_w * 0.75))
        cy = float(np.clip(cy, out_h * 0.25, out_h * 0.75))
        K_grd = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        # 7) poses
        sat_c2w = self._build_sat_c2w()
        grd_c2w = self._build_grd_c2w(gt_shift_x, gt_shift_y, theta_deg)

        # 8) GT projection of ground camera into sat pixel plane (matches
        # the local-frame formula used by eval_sat.py and Pi3KittiLoss):
        #   P_local = R_sat^T (T_grd - T_sat)
        #   u = P_local[0]/mpp + W/2 ; v = P_local[1]/mpp + H/2
        delta = grd_c2w[:3, 3] - sat_c2w[:3, 3]
        P_local_gt = sat_c2w[:3, :3].T @ delta
        u_gt = float(P_local_gt[0] / sat_mpp_final + out_w / 2.0)
        v_gt = float(P_local_gt[1] / sat_mpp_final + out_h / 2.0)

        # 9) dummy depthmaps so BaseDataset's valid_mask assertion passes.
        # Use 1 m for ground, h_sat for sat.  These never enter the KITTI
        # loss (it ignores all point terms), they only fill the data
        # contract.  Use a small positive constant <500 so the sat
        # ``new_depthmap < 500`` mask still contains valid pixels.
        depth_grd = np.full((out_h, out_w), 1.0, dtype=np.float32)
        depth_sat = np.full((out_h, out_w), self.sat_height, dtype=np.float32)

        sat_view = dict(
            img=sat_arr,
            depthmap=depth_sat,
            camera_pose=sat_c2w.astype(np.float32),
            camera_intrinsics=K_sat.astype(np.float32),
            sat_gap=self.sat_height,
            dataset=self.dataset_label,
            label=f"satellite_{file_name}",
            instance=f"sat_{index}",
            sat_meters=float(self.satmap_side * self.meter_per_pixel),
            sat_shift_east=0.0,
            sat_shift_south=0.0,
            is_satellite=True,
            kitti_sat_mpp=float(sat_mpp_final),
            kitti_ground_uv_in_sat=np.array([u_gt, v_gt], dtype=np.float32),
        )
        grd_view = dict(
            img=grd_arr,
            depthmap=depth_grd,
            camera_pose=grd_c2w.astype(np.float32),
            camera_intrinsics=K_grd.astype(np.float32),
            sat_gap=self.sat_height,
            dataset=self.dataset_label,
            label=f"ground_{file_name}",
            instance=f"grd_{index}",
            sat_meters=0.0,
            sat_shift_east=0.0,
            sat_shift_south=0.0,
            is_satellite=False,
            kitti_sat_mpp=float(sat_mpp_final),
            kitti_ground_uv_in_sat=np.array([u_gt, v_gt], dtype=np.float32),
        )
        # Run the BaseDataset crop+resize pipeline so depth, intrinsics and
        # image resolution land at the requested target.
        for view in (sat_view, grd_view):
            img, dep, K = self._crop_resize_if_necessary(
                view['img'], view['depthmap'], view['camera_intrinsics'],
                resolution, rng=rng, info=view['label'],
                sat=view['is_satellite'],
            )
            view['img'] = img
            view['depthmap'] = dep
            view['camera_intrinsics'] = K

        return {"satellite": [sat_view], "ground_drone": [grd_view]}
