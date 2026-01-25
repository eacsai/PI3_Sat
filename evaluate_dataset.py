import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定使用的GPU设备ID
import sys
sys.path.append('.')

import torch
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from omegaconf import OmegaConf
import hydra
from copy import deepcopy

from pi3.utils.basic import load_images_as_tensor
from pi3.utils.geometry import depth_edge, se3_inverse, homogenize_points
from pi3.utils.alignment import align_points_scale
from pi3.models.pi3_training import Pi3
from datasets import create_dataloader
from utils.misc import move_to_device, get_logger, pretty_print_hydra_config, is_logging_process
from utils.dist import MetricLogger
from easydict import EasyDict
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict


def compute_quality_metrics(res, batch, view_idx=0):
    """
    计算图片质量的评估指标，与 GT 做比较

    Args:
        res: 模型输出结果，包含 points, local_points, conf, camera_poses
        batch: 数据批次（从 dataloader 获取）
        view_idx: 视图索引，默认为 0

    Returns:
        dict: 包含各项质量指标的字典
    """
    import torch.nn.functional as F

    # res 的 shape: [B, N, H, W, ...] 其中 B=1 (batch), N=frame_num
    # 先移除 batch 维度
    pred_local_pts = res['local_points'][0]  # [N, H, W, 3]
    pred_pts = res['points'][0]  # [N, H, W, 3]
    pred_poses = res['camera_poses'][0]  # [N, 4, 4]
    conf = res['conf'][0] if res['conf'] is not None else None  # [N, H, W, 1] or None

    # 从 batch 中提取 GT 数据
    N = pred_local_pts.shape[0]
    gt_pts = torch.stack([batch[i]['pts3d'] for i in range(N)], dim=0).to(pred_local_pts.device)  # [N, H, W, 3]
    gt_valid_mask = torch.stack([batch[i]['valid_mask'] for i in range(N)], dim=0).to(pred_local_pts.device)  # [N, H, W]
    gt_poses = torch.stack([batch[i]['camera_pose'] for i in range(N)], dim=0).to(pred_local_pts.device)  # [N, 4, 4]

    # 预测置信度（如果有）
    if conf is not None:
        conf_scores = torch.sigmoid(conf[..., 0])  # [N, H, W]
        conf_mean = conf_scores[gt_valid_mask].mean().item() if gt_valid_mask.sum() > 0 else 0.0
    else:
        conf_mean = 0.0

    # ========== 与 GT 对齐并计算误差 ==========
    # 1. 归一化 GT（参考 Pi3Loss.prepare_gt）
    # transform to first frame camera coordinate
    w2c_target = se3_inverse(gt_poses[0:1])  # [1, 4, 4]
    gt_pts_normalized = torch.einsum('ij, nhwj -> nhwi', w2c_target, homogenize_points(gt_pts))[..., :3]
    gt_poses_normalized = torch.einsum('ij, nhjk -> nhik', w2c_target, gt_poses)

    # normalize by scene depth
    valid_batch = gt_valid_mask.sum() > 0
    if valid_batch:
        all_gt_pts = gt_pts_normalized.clone()
        all_gt_pts[~gt_valid_mask] = 0
        all_dis = all_gt_pts.norm(dim=-1)
        norm_factor = all_dis.sum() / (gt_valid_mask.float().sum() + 1e-8)
        gt_pts_normalized = gt_pts_normalized / norm_factor
        gt_poses_normalized[..., :3, 3] /= norm_factor

    # GT local points (相机坐标系下的点)
    gt_local_pts = torch.einsum('nij, nhwj -> nhwi', se3_inverse(gt_poses_normalized), homogenize_points(gt_pts_normalized))[..., :3]

    # 2. 对齐预测结果并计算 scale
    # 使用 ROE (Region of Interest) 对齐
    def prepare_ROE(pts, mask, target_size=4096):
        valid_pts = pts[mask]
        if valid_pts.shape[0] > 0:
            valid_pts = valid_pts.permute(1, 0).unsqueeze(0)  # (1, 3, N1)
            valid_pts = F.interpolate(valid_pts, size=target_size, mode='nearest')  # (1, 3, target_size)
            valid_pts = valid_pts.squeeze(0).permute(1, 0)  # (target_size, 3)
        else:
            valid_pts = torch.ones((target_size, 3), device=pts.device)
        return valid_pts

    xyz_pred = prepare_ROE(pred_local_pts.reshape(-1, 3), gt_valid_mask.reshape(-1))
    xyz_gt = prepare_ROE(gt_local_pts.reshape(-1, 3), gt_valid_mask.reshape(-1))
    xyz_weights = prepare_ROE((1.0 / (gt_local_pts[..., 2].clamp(min=0.1) + 1e-6)).reshape(-1, 1), gt_valid_mask.reshape(-1))[:, 0]

    scale = align_points_scale(xyz_pred.unsqueeze(0), xyz_gt.unsqueeze(0), xyz_weights.unsqueeze(0))[0]
    scale = scale.clamp(min=1e-6)

    # 3. 计算各种误差指标
    aligned_pred_local = scale * pred_local_pts

    # ========== 深度误差 ==========
    pred_depth = pred_local_pts[..., 2]  # [N, H, W]
    gt_depth = gt_local_pts[..., 2]  # [N, H, W]

    # 只在有效区域计算
    valid_mask = gt_valid_mask & (gt_depth > 0) & (pred_depth > 0)

    if valid_mask.sum() > 0:
        depth_abs_err = (pred_depth[valid_mask] - gt_depth[valid_mask]).abs()
        depth_mae = depth_abs_err.mean().item()
        depth_rmse = torch.sqrt((pred_depth[valid_mask] - gt_depth[valid_mask])**2).mean().item()

        # 相对误差
        rel_err = ((pred_depth[valid_mask] - gt_depth[valid_mask]).abs() / gt_depth[valid_mask])
        depth_rel_mae = rel_err.mean().item()
    else:
        depth_mae = depth_rmse = depth_rel_mae = 0.0

    # ========== 点云误差 ==========
    if valid_mask.sum() > 0:
        pts_l2_err = torch.norm(aligned_pred_local[valid_mask] - gt_local_pts[valid_mask], dim=-1)
        pts_mae = pts_l2_err.mean().item()
    else:
        pts_mae = 0.0

    # ========== 相机位姿误差 ==========
    # 归一化预测位姿
    pred_poses_normalized = pred_poses.clone()
    pred_poses_normalized[..., :3, 3] *= scale

    # 转换到第一帧坐标系
    pred_w2c = se3_inverse(pred_poses_normalized)
    gt_w2c = se3_inverse(gt_poses_normalized)

    # 计算相对位姿误差 (相对于第一帧)
    # 第一帧误差应该接近 0
    rot_err_first = rotation_angle_error(pred_w2c[0, :3, :3], gt_w2c[0, :3, :3])
    trans_err_first = (pred_w2c[0, :3, 3] - gt_w2c[0, :3, 3]).norm().item()

    # 计算所有帧的平均误差
    rot_errors = []
    trans_errors = []
    for i in range(N):
        rot_err = rotation_angle_error(pred_w2c[i, :3, :3], gt_w2c[i, :3, :3])
        trans_err = (pred_w2c[i, :3, 3] - gt_w2c[i, :3, 3]).norm()
        rot_errors.append(rot_err)
        trans_errors.append(trans_err)

    rot_mae = torch.stack(rot_errors).mean().item()
    trans_mae = torch.stack(trans_errors).mean().item()

    # ========== 其他统计指标 ==========
    num_pixels = valid_mask.numel()
    num_valid_points = valid_mask.sum().item()
    valid_ratio = num_valid_points / num_pixels if num_pixels > 0 else 0.0

    # 深度统计
    if valid_mask.sum() > 0:
        depth_mean = pred_depth[valid_mask].mean().item()
        depth_std = pred_depth[valid_mask].std().item()
        depth_min = pred_depth[valid_mask].min().item()
        depth_max = pred_depth[valid_mask].max().item()
    else:
        depth_mean = depth_std = depth_min = depth_max = 0.0

    # 点云范围
    if valid_mask.sum() > 0:
        pts = pred_pts[valid_mask]
        points_range = (pts.max(dim=0)[0] - pts.min(dim=0)[0]).cpu().numpy()
    else:
        points_range = np.array([0.0, 0.0, 0.0])

    # 从 batch 中获取元数据
    batch_size = len(batch[view_idx]['idx'])
    dataset_names = batch[view_idx].get('dataset', ['unknown'] * batch_size)
    scene_labels = batch[view_idx].get('label', ['unknown'] * batch_size)
    instance_ids = batch[view_idx].get('instance', ['unknown'] * batch_size)

    metrics = {
        # 基本统计
        'num_valid_points': int(num_valid_points),
        'num_total_pixels': int(num_pixels),
        'valid_point_ratio': float(valid_ratio),
        'scale': float(scale),

        # 深度误差
        'depth_mae': float(depth_mae),
        'depth_rmse': float(depth_rmse),
        'depth_rel_mae': float(depth_rel_mae),

        # 点云误差
        'pts_mae': float(pts_mae),

        # 相机位姿误差
        'rot_mae': float(rot_mae),  # 弧度
        'rot_mae_deg': float(np.degrees(rot_mae)),
        'trans_mae': float(trans_mae),
        'rot_err_first_deg': float(np.degrees(rot_err_first)),
        'trans_err_first': float(trans_err_first),

        # 深度统计
        'depth_mean': float(depth_mean),
        'depth_std': float(depth_std),
        'depth_min': float(depth_min),
        'depth_max': float(depth_max),

        # 点云范围
        'points_range_x': float(points_range[0]),
        'points_range_y': float(points_range[1]),
        'points_range_z': float(points_range[2]),

        # 置信度
        'confidence_mean': float(conf_mean),

        # 综合质量分数 (越小越好，与误差相关)
        'quality_score': float(depth_mae + pts_mae + rot_mae + trans_mae),

        # 元数据
        'batch_size': batch_size,
        'dataset_names': dataset_names if isinstance(dataset_names, list) else list(dataset_names),
        'scene_labels': scene_labels if isinstance(scene_labels, list) else list(scene_labels),
        'instance_ids': instance_ids if isinstance(instance_ids, list) else list(instance_ids),
    }

    return metrics


def rotation_angle_error(R, R_gt, eps=1e-6):
    """计算旋转矩阵的角度误差"""
    residual = torch.matmul(R.transpose(0, 1), R_gt)
    trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum()
    cosine = (trace - 1) / 2
    return torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))


def evaluate_dataset(model, test_loader, device, dtype, logger, output_file=None, max_batches=None, base_seed=666):
    """
    评估整个数据集

    Args:
        model: Pi3 模型
        test_loader: 测试数据加载器
        device: 计算设备
        dtype: 数据类型
        logger: 日志记录器
        output_file: 结果保存路径
        max_batches: 最大评估批次数（用于快速测试）
        base_seed: 基础随机种子
    """
    model.eval()

    # 设置 epoch（与 trainer 保持一致）
    if hasattr(test_loader, 'dataset') and hasattr(test_loader.dataset, 'set_epoch'):
        test_loader.dataset.set_epoch(0, base_seed=base_seed)
    if hasattr(test_loader, 'batch_sampler') and hasattr(test_loader.batch_sampler, 'batch_sampler') and hasattr(test_loader.batch_sampler.batch_sampler, 'sampler') and hasattr(test_loader.batch_sampler.batch_sampler.sampler, 'set_epoch'):
        test_loader.batch_sampler.batch_sampler.sampler.set_epoch(0, base_seed=base_seed)
    if hasattr(test_loader, 'batch_sampler') and hasattr(test_loader.batch_sampler, 'set_epoch'):
        test_loader.batch_sampler.set_epoch(0, base_seed=base_seed)

    all_results = []
    all_quality_scores = []
    scene_results = {}  # 按 scene 聚合结果

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluating:"

    logger.info(f"Start evaluation on test dataset...")
    if max_batches:
        logger.info(f"Limiting evaluation to {max_batches} batches")

    with torch.no_grad():
        for batch_idx, batch in enumerate(metric_logger.log_every(
            test_loader, print_freq=10, header=header
        )):
            if max_batches is not None and batch_idx >= max_batches:
                break

            try:
                batch = move_to_device(batch, device)

                # 从 batch 中提取图像
                # batch 是一个 list，每个元素对应一个 view
                imgs = batch[0]['img']  # (N, 3, H, W)
                B, C, H, W = imgs.shape

                # 推理
                with torch.amp.autocast('cuda', dtype=dtype):
                    res = model(imgs[None])  # 添加 batch 维度

                # 计算指标
                metrics = compute_quality_metrics(res, batch, view_idx=0)

                # 添加批次索引
                metrics['batch_idx'] = batch_idx

                # 按场景聚合结果
                for i in range(metrics['batch_size']):
                    scene_name = metrics['scene_labels'][i]
                    dataset_name = metrics['dataset_names'][i]
                    instance_id = metrics['instance_ids'][i]
                    key = f"{dataset_name}/{scene_name}/{instance_id}"

                    if key not in scene_results:
                        scene_results[key] = {
                            'dataset': dataset_name,
                            'scene': scene_name,
                            'instance': instance_id,
                            'count': 0,
                            'quality_scores': [],
                        }

                    scene_results[key]['count'] += 1
                    scene_results[key]['quality_scores'].append(metrics['quality_score'])

                all_results.append(metrics)
                all_quality_scores.append(metrics['quality_score'])

                metric_logger.update(
                    quality_score=metrics['quality_score'],
                    valid_ratio=metrics['valid_point_ratio'],
                    depth_mae=metrics['depth_mae'],
                    pts_mae=metrics['pts_mae'],
                    rot_mae_deg=metrics['rot_mae_deg'],
                    trans_mae=metrics['trans_mae'],
                )

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # 计算统计信息
    if all_quality_scores:
        # 计算所有误差指标的平均值
        all_depth_mae = [r['depth_mae'] for r in all_results]
        all_depth_rmse = [r['depth_rmse'] for r in all_results]
        all_depth_rel_mae = [r['depth_rel_mae'] for r in all_results]
        all_pts_mae = [r['pts_mae'] for r in all_results]
        all_rot_mae = [r['rot_mae'] for r in all_results]
        all_trans_mae = [r['trans_mae'] for r in all_results]
        all_rot_mae_deg = [r['rot_mae_deg'] for r in all_results]

        stats = {
            'total_batches': len(all_results),
            'total_samples': sum(r['batch_size'] for r in all_results),
            'quality_score_mean': float(np.mean(all_quality_scores)),
            'quality_score_std': float(np.std(all_quality_scores)),
            'quality_score_min': float(np.min(all_quality_scores)),
            'quality_score_max': float(np.max(all_quality_scores)),
            'quality_score_median': float(np.median(all_quality_scores)),
            # 误差统计
            'depth_mae_mean': float(np.mean(all_depth_mae)),
            'depth_rmse_mean': float(np.mean(all_depth_rmse)),
            'depth_rel_mae_mean': float(np.mean(all_depth_rel_mae)),
            'pts_mae_mean': float(np.mean(all_pts_mae)),
            'rot_mae_deg_mean': float(np.mean(all_rot_mae_deg)),
            'trans_mae_mean': float(np.mean(all_trans_mae)),
        }

        # 计算每个场景的平均指标
        scene_stats = []
        for key, data in scene_results.items():
            scores = data['quality_scores']
            scene_stats.append({
                'key': key,
                'dataset': data['dataset'],
                'scene': data['scene'],
                'instance': data['instance'],
                'count': data['count'],
                'quality_score_mean': float(np.mean(scores)),
                'quality_score_std': float(np.std(scores)) if len(scores) > 1 else 0.0,
                'quality_score_min': float(np.min(scores)),
                'quality_score_max': float(np.max(scores)),
            })

        # 按平均质量分数排序（越小越好，所以 reverse=False）
        scene_stats_sorted = sorted(scene_stats, key=lambda x: x['quality_score_mean'])

        # 打印统计结果
        print("\n" + "="*60)
        print("数据集评估结果统计")
        print("="*60)
        print(f"总批次数: {stats['total_batches']}")
        print(f"总样本数: {stats['total_samples']}")
        print(f"唯一场景数: {len(scene_stats)}")

        print("\n" + "-"*60)
        print("误差指标 (越小越好)")
        print("-"*60)
        print(f"  深度 MAE:     {stats['depth_mae_mean']:.4f}")
        print(f"  深度 RMSE:    {stats['depth_rmse_mean']:.4f}")
        print(f"  深度相对MAE:  {stats['depth_rel_mae_mean']:.4f}")
        print(f"  点云 MAE:     {stats['pts_mae_mean']:.4f}")
        print(f"  旋转 MAE:     {stats['rot_mae_deg_mean']:.2f}°")
        print(f"  平移 MAE:     {stats['trans_mae_mean']:.4f}")

        print("\n" + "-"*60)
        print("综合质量分数统计")
        print("-"*60)
        print(f"  平均值: {stats['quality_score_mean']:.4f}")
        print(f"  中位数: {stats['quality_score_median']:.4f}")
        print(f"  标准差: {stats['quality_score_std']:.4f}")
        print(f"  最小值: {stats['quality_score_min']:.4f}")
        print(f"  最大值: {stats['quality_score_max']:.4f}")
        print("="*60)

        # 按数据集聚合
        dataset_stats = {}
        for s in scene_stats:
            ds = s['dataset']
            if ds not in dataset_stats:
                dataset_stats[ds] = []
            dataset_stats[ds].append(s['quality_score_mean'])

        print("\n按数据集统计:")
        for ds, scores in sorted(dataset_stats.items()):
            print(f"  {ds}: {np.mean(scores):.4f} (±{np.std(scores):.4f}), n={len(scores)}")

        # 显示最佳和最差场景 (误差越小越好)
        print("\n最佳质量场景/最小误差 (Top 10):")
        for i, s in enumerate(scene_stats_sorted[:10], 1):
            print(f"  {i:2d}. {s['key']}: {s['quality_score_mean']:.4f} (n={s['count']})")

        print("\n最差质量场景/最大误差 (Bottom 10):")
        for i, s in enumerate(reversed(scene_stats_sorted[-10:]), 1):
            print(f"  {i:2d}. {s['key']}: {s['quality_score_mean']:.4f} (n={s['count']})")
        print("="*60)

        # 保存结果到文件
        output = {
            'statistics': stats,
            'dataset_statistics': {
                ds: {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'count': len(scores),
                }
                for ds, scores in dataset_stats.items()
            },
            'scene_results': scene_stats_sorted,
            'batch_results': all_results,
        }

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"\n结果已保存到: {output_file}")
        else:
            # 默认保存到 outputs 目录
            default_output = Path('outputs/evaluation_results.json')
            default_output.parent.mkdir(parents=True, exist_ok=True)
            with open(default_output, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"\n结果已保存到: {default_output}")

        return stats

    else:
        logger.warning("没有成功评估任何批次")
        return None


@hydra.main(version_base="1.2", config_path="./configs", config_name="megadepth")
def main(hydra_cfg):
    # 添加 job_logging_cfg（与 trainer 保持一致）
    with open_dict(hydra_cfg):
        hydra_cfg.job_logging_cfg = HydraConfig.get().job_logging

    # 设置日志
    logger = get_logger(hydra_cfg, os.path.basename(__file__))
    if is_logging_process():
        pretty_print_hydra_config(hydra_cfg)

    # 解析命令行参数
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ckpt", type=str, default='outputs/template/ckpts/best_model',
                        help="Path to the model checkpoint file. Default: None (use pretrained)")
    parser.add_argument("--output_file", type=str, default='outputs/template/eval.json',
                        help="Path to save the evaluation results JSON file.")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Maximum number of batches to evaluate (for quick testing).")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")

    args, unknown = parser.parse_known_args()
    device = torch.device(args.device)

    # 1. 准备模型
    logger.info("Loading model...")
    if args.ckpt is not None:
        model = Pi3().to(device).eval()

        # 处理 checkpoint 目录（accelerator.save_state 保存的格式）
        ckpt_path = Path(args.ckpt)
        if ckpt_path.is_dir():
            # 查找目录中的模型文件
            model_file = None
            for fname in ['pytorch_model.bin', 'model.safetensors', 'model.pth']:
                fpath = ckpt_path / fname
                if fpath.exists():
                    model_file = str(fpath)
                    break

            if model_file is None:
                raise FileNotFoundError(
                    f"Cannot find model file in checkpoint directory {args.ckpt}. "
                    f"Expected: pytorch_model.bin, model.safetensors, or model.pth"
                )

            logger.info(f"Found model file in checkpoint directory: {model_file}")
            ckpt_path = Path(model_file)

        if str(ckpt_path).endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(str(ckpt_path))
        else:
            weight = torch.load(str(ckpt_path), map_location=device, weights_only=False)

        model.load_state_dict(weight)
        logger.info(f"Loaded model weights from: {ckpt_path}")
    else:
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
        logger.info("Loaded pretrained model from HuggingFace")

    # 2. 确定数据类型
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    logger.info(f"Using dtype: {dtype}")

    # 3. 创建测试数据加载器
    logger.info("Creating test dataloader...")
    test_loader = create_dataloader(hydra_cfg, 'test')
    logger.info(f"Test dataloader created, dataset length: {len(test_loader.dataset)}")

    # 4. 评估数据集
    stats = evaluate_dataset(
        model=model,
        test_loader=test_loader,
        device=device,
        dtype=dtype,
        logger=logger,
        output_file=args.output_file,
        max_batches=args.max_batches,
        base_seed=hydra_cfg.train.base_seed,
    )

    logger.info("Evaluation completed!")


if __name__ == '__main__':
    main()
