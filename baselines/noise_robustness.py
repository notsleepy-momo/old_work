"""
实验组 B：噪声鲁棒性 — noise_robustness.py

UbiComp robustness evaluation: 三组实验，v2 (review改进版)

  B1. 设备坐标扰动 — correlated perturbation (全局偏移 + 局部噪声)
      扰动等级: 0dm, 5dm, 10dm, 20dm
      输出: CLS / RTA / F1 曲线 + Relative Retention
      目标: 10dm 内性能稳定

  B2. 行为轨迹稀疏度 — temporal downsampling + multi-seed averaging
      数据量: 每隔N点采样 (模拟低采样率) + 连续时间段缺失
      输出: mean ± std over 5 seeds

  B3. 图像退化 — 扫地机器人APP截图真实退化
      - 分辨率缩放 (不同手机屏幕)
      - 地图缺角 (机器人未扫完全部房间)
      - SLAM 墙线扰动 (建图误差导致墙壁扭曲)
      输出: F1 下降幅度 / 小家具 recall

用法:
  # 全部 3 组 (CV-only)
  python baselines/noise_robustness.py --code 0622 --skip-seg --skip-agents

  # 完整 LLM 模式
  python baselines/noise_robustness.py --code 0622 --skip-seg
"""
import os, sys, argparse, yaml, json, shutil, copy, random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

# ── Windows 下强制 UTF-8 输出 ──
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── 加载 .env ──
from dotenv import load_dotenv
_env_path = os.path.join(PROJECT_ROOT, '.env')
if os.path.exists(_env_path):
    loaded = False
    for enc in ('utf-8-sig', 'utf-16-le', 'utf-16'):
        try:
            load_dotenv(dotenv_path=_env_path, encoding=enc)
            loaded = True
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if not loaded:
        load_dotenv(dotenv_path=_env_path, encoding='utf-8', errors='ignore')
    print(f"[dotenv] 已加载: {_env_path}")

from pipeline import Pipeline
from evaluation import evaluate_single

# ============================================================
# 辅助函数
# ============================================================

def perturb_device_yaml_correlated(
    src_path: str, dst_path: str, noise_dm: float, seed: int = 42
):
    """B1 v2: 对智能设备坐标添加 correlated 扰动 (全局偏移 + 局部噪声)。

    真实 calibration error 通常表现为整体偏移 + 小量局部误差，
    而非每个设备独立随机跳跃。这样可以避免不现实的 topology distortion。

    Args:
        src_path: 原始 smartDevice YAML 路径
        dst_path: 输出路径
        noise_dm: 噪声幅度 (dm)。全局偏移 σ_global = noise_dm/3, 局部 σ_local = noise_dm/9
        seed: 随机种子
    """
    rng = np.random.RandomState(seed)
    with open(src_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    sigma_global = noise_dm / 3.0  # ±N dm → 全局 σ = N/3
    sigma_local = noise_dm / 9.0   # 局部误差为全局的 1/3

    global_dx = rng.randn() * sigma_global
    global_dy = rng.randn() * sigma_global

    for fur in data.get('house', {}).get('furniture', []):
        pos = fur.get('position', {})
        if 'x' in pos:
            pos['x'] = round(float(pos['x']) + global_dx + rng.randn() * sigma_local, 2)
        if 'y' in pos:
            pos['y'] = round(float(pos['y']) + global_dy + rng.randn() * sigma_local, 2)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=None, allow_unicode=True, sort_keys=False, indent=2)
    return dst_path


def temporal_downsample_trajectory(
    src_path: str, dst_path: str, step: int, seed: int = 42
):
    """B2 v2: 时序降采样 —— 每隔 step 个点取一个 (模拟低采样率)。

    Args:
        src_path: 原始 trajectory JSON
        dst_path: 输出路径
        step: 采样步长 (step=1 → 全量, step=2 → 50%, step=4 → 25%, ...)
        seed: 随机种子 (用于随机起始偏移)
    """
    with open(src_path, 'r', encoding='utf-8') as f:
        traj_data = json.load(f)

    rng = np.random.RandomState(seed)
    start_offset = rng.randint(0, step)
    subsampled = traj_data[start_offset::step]

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, 'w', encoding='utf-8') as f:
        json.dump(subsampled, f, indent=2, ensure_ascii=False)
    return dst_path, len(traj_data), len(subsampled)


def day_truncate_trajectory(
    src_path: str, dst_path: str, day_fraction: float
):
    """B2 v2: 按时间线截断 —— 只保留前 day_fraction 时间段的轨迹 (模拟短周期采集)。

    Args:
        src_path: 原始 trajectory JSON
        dst_path: 输出路径
        day_fraction: 保留的时间比例 (0~1)
    """
    with open(src_path, 'r', encoding='utf-8') as f:
        traj_data = json.load(f)

    if not traj_data:
        return dst_path, 0, 0

    # 按 start_time 排序
    sorted_data = sorted(traj_data, key=lambda x: x.get('start_time', ''))
    n_total = len(sorted_data)
    n_keep = max(1, int(n_total * day_fraction))
    truncated = sorted_data[:n_keep]

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, 'w', encoding='utf-8') as f:
        json.dump(truncated, f, indent=2, ensure_ascii=False)
    return dst_path, n_total, n_keep


def rescale_image(src_path: str, dst_path: str, scale: float = 0.5):
    """B3: 分辨率缩放 —— 模拟不同手机屏幕分辨率。

    先缩小再拉回原尺寸，模拟低分辨率截图放大后的模糊效果。

    Args:
        src_path: 原始图像路径
        dst_path: 输出路径
        scale: 缩放比例 (0~1, 越小越退化)
    """
    import cv2
    img = cv2.imread(src_path)
    if img is None:
        print(f"  [警告] 无法读取图像: {src_path}, 跳过")
        return src_path
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    degraded = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cv2.imwrite(dst_path, degraded)
    return dst_path


def crop_image(src_path: str, dst_path: str, crop_fraction: float = 0.2):
    """B3: 地图缺角 —— 模拟机器人未完整扫描。

    从图像边缘裁掉 crop_fraction 比例的区域（右侧+底部各裁一半），
    剩余部分拉伸回原尺寸。

    Args:
        src_path: 原始图像路径
        dst_path: 输出路径
        crop_fraction: 裁剪比例 (0~1)
    """
    import cv2
    img = cv2.imread(src_path)
    if img is None:
        print(f"  [警告] 无法读取图像: {src_path}, 跳过")
        return src_path
    h, w = img.shape[:2]
    crop_r = int(w * crop_fraction / 2)
    crop_b = int(h * crop_fraction / 2)
    cropped = img[0:h - crop_b, 0:w - crop_r]
    degraded = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cv2.imwrite(dst_path, degraded)
    return dst_path


def add_wall_noise(src_path: str, dst_path: str, sigma: float = 1.5):
    """B3: SLAM 墙线扰动 —— 模拟建图误差导致墙壁轻微扭曲。

    对图像施加小幅度随机位移场（elastic deformation 的简化版），
    仅对亮区域（墙/hatch）产生可见影响。

    Args:
        src_path: 原始图像路径
        dst_path: 输出路径
        sigma: 位移场标准差 (控制扭曲程度)
    """
    import cv2
    img = cv2.imread(src_path)
    if img is None:
        print(f"  [警告] 无法读取图像: {src_path}, 跳过")
        return src_path
    h, w = img.shape[:2]

    # 生成平滑随机位移场
    rng = np.random.RandomState(42)
    grid = min(h, w) // 20  # 控制点间距
    gh, gw = h // grid + 2, w // grid + 2
    dx_small = rng.randn(gh, gw).astype(np.float32) * sigma
    dy_small = rng.randn(gh, gw).astype(np.float32) * sigma

    dx = cv2.resize(dx_small, (w, h), interpolation=cv2.INTER_LINEAR)
    dy = cv2.resize(dy_small, (w, h), interpolation=cv2.INTER_LINEAR)

    map_x = (np.arange(w, dtype=np.float32).reshape(1, w) + dx).astype(np.float32)
    map_y = (np.arange(h, dtype=np.float32).reshape(h, 1) + dy).astype(np.float32)

    degraded = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cv2.imwrite(dst_path, degraded)
    return dst_path


# ============================================================
# Pipeline 运行封装
# ============================================================

def run_pipeline_with_inputs(
    code: str, variant_name: str, output_dir: str,
    image_path: str, smart_device_path: str, trajectory_path: str,
    hatch_mask_path: str = None,
    skip_seg: bool = False,
    skip_agents: bool = False,
    api_key: str = None, base_url: str = None,
) -> str:
    """运行 Pipeline 并返回最终 YAML 路径"""
    print(f"\n  {'─'*50}")
    print(f"  Variant: {variant_name}")

    pipeline = Pipeline(api_key=api_key, base_url=base_url, ablation={}, code=code)
    if skip_agents:
        pipeline.ablation['skip_llm_correction'] = True

    safe_name = variant_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    variant_out_dir = os.path.join(output_dir, f"noise_{code}_{safe_name}")
    os.makedirs(variant_out_dir, exist_ok=True)

    yaml_dir = os.path.join(variant_out_dir, 'yaml')
    os.makedirs(yaml_dir, exist_ok=True)

    # ── Step 1: 房间分割 ──
    if skip_seg:
        seg_path = os.path.join(PROJECT_ROOT, 'output', code, 'yaml', f'01_seg_pixel_{code}.yaml')
        if not os.path.exists(seg_path):
            raise FileNotFoundError(f"skip_seg=True but no existing YAML: {seg_path}")
        dst_seg = os.path.join(yaml_dir, f'01_seg_pixel_{code}.yaml')
        shutil.copy2(seg_path, dst_seg)
        seg_path = dst_seg
        print(f"  [skip_seg] 复用分割: {seg_path}")
    else:
        seg_path = pipeline.run_seg_step(image_path, variant_out_dir)
        src_yaml = os.path.join(variant_out_dir, 'floorplan_real.yaml')
        dst_yaml = os.path.join(yaml_dir, f'01_seg_pixel_{code}.yaml')
        if os.path.exists(src_yaml):
            shutil.move(src_yaml, dst_yaml)
            with open(dst_yaml, 'r', encoding='utf-8') as f:
                seg_layout = yaml.safe_load(f)
            for room in seg_layout.get('house', {}).get('rooms', []):
                if 'id' not in room and 'name' in room:
                    room['id'] = room['name']
            with open(dst_yaml, 'w', encoding='utf-8') as f:
                yaml.dump(seg_layout, f, default_flow_style=None,
                          allow_unicode=True, sort_keys=False, indent=2)
            seg_path = dst_yaml

    # ── Step 2: 家具检测 + 坐标统一 ──
    greyroom_path = os.path.join(variant_out_dir, 'masks', 'greyroom.png')
    hm_path = hatch_mask_path
    if hm_path and os.path.exists(hm_path):
        pass
    else:
        default_hm = os.path.join(variant_out_dir, 'masks', 'hatch_mask.png')
        if os.path.exists(default_hm):
            hm_path = default_hm

    seg_image = image_path
    if os.path.exists(greyroom_path):
        seg_image = greyroom_path

    world_yaml_path, trajectory_data = pipeline.step2_furniture_and_devices(
        seg_image, seg_path, trajectory_path,
        smart_device_path, hm_path, variant_out_dir)

    dst_02 = os.path.join(yaml_dir, f'02_furniture_world_{code}.yaml')
    if os.path.exists(world_yaml_path) and world_yaml_path != dst_02:
        shutil.copy2(world_yaml_path, dst_02)
        world_yaml_path = dst_02

    # ── Steps 3-5: LLM Agents ──
    if skip_agents:
        logger_ = __import__('logging').getLogger('pipeline')
        logger_.info("=== LLM Agents 全部跳过 (--skip-agents) ===")
        final_path = os.path.join(yaml_dir, f'03_final_{code}.yaml')
        shutil.copy2(world_yaml_path, final_path)
    else:
        room_analyses = pipeline.room_analysis_step(world_yaml_path, trajectory_data)
        behavior_analyses = pipeline.behavior_analysis_step(
            world_yaml_path, trajectory_data, room_analyses)
        final_path = pipeline.step5_furniture_naming_and_merge(
            world_yaml_path, room_analyses, behavior_analyses, variant_out_dir)

    dst_final = os.path.join(yaml_dir, f'03_final_{code}.yaml')
    if os.path.exists(final_path) and final_path != dst_final:
        shutil.copy2(final_path, dst_final)
        final_path = dst_final

    print(f"  Output: {final_path}")
    return dst_final


# ============================================================
# B1: 设备坐标扰动 (v2: correlated perturbation)
# ============================================================

def run_b1_device_perturbation(
    code: str, image_path: str, smart_device_path: str,
    trajectory_path: str, hatch_mask_path: str,
    output_dir: str, gt_path: str, skip_seg: bool,
    skip_agents: bool = False,
    api_key: str = None, base_url: str = None,
) -> Dict:
    """B1 v2: 设备坐标扰动 — correlated perturbation + multi-seed averaging.

    perturbation_levels 定义为 noise_dm (σ_global = noise_dm/3, σ_local = noise_dm/9)。
    每个非零扰动等级在 MULTI_SEEDS 上取 mean±std，消除单次随机抽取的波动。
    """
    print(f"\n{'#'*60}")
    print(f"#  B1: 设备坐标扰动 — Correlated Perturbation (multi-seed)")
    print(f"#  σ_global = noise/3, σ_local = noise/9, seeds={MULTI_SEEDS}")
    print(f"{'#'*60}")

    def _mean_std(values):
        arr = np.array(values)
        return float(np.mean(arr)), float(np.std(arr, ddof=1))

    perturbation_levels = [0, 2, 5, 10, 15, 20]  # noise_dm
    temp_dir = os.path.join(output_dir, '_temp')
    os.makedirs(temp_dir, exist_ok=True)

    results = {}
    cls0 = None  # baseline CLS，用于 per-seed retention

    for level in perturbation_levels:
        sigma_global = level / 3.0
        variant_name = f"B1_Correlated_σg{sigma_global:.2f}dm"
        try:
            if level == 0:
                existing_final = os.path.join(
                    PROJECT_ROOT, 'output', code, 'yaml', f'03_final_{code}.yaml')
                if os.path.exists(existing_final):
                    print(f"  [复用] σ_g=0dm 已有结果: {existing_final}")
                    pred_path = existing_final
                else:
                    pred_path = run_pipeline_with_inputs(
                        code, variant_name, output_dir,
                        image_path, smart_device_path, trajectory_path,
                        hatch_mask_path, skip_seg, skip_agents, api_key, base_url)

                result_a, result_b = evaluate_single(pred_path, gt_path)
                r = result_a
                cls0 = r.cls
                results[level] = {
                    "result_a": result_a,
                    "result_b": result_b,
                    "pred_path": pred_path,
                    "sigma_global": 0.0,
                    "multi_seed": False,
                }
            else:
                # ── multi-seed 运行 ──
                all_metrics = []
                for seed in MULTI_SEEDS:
                    perturbed_device_path = os.path.join(
                        temp_dir, f'smartDevice_{code}_correlated_{level}dm_s{seed}.yaml')
                    perturb_device_yaml_correlated(
                        smart_device_path, perturbed_device_path, level, seed)

                    pred_path = run_pipeline_with_inputs(
                        code, f"{variant_name}_s{seed}", output_dir,
                        image_path, perturbed_device_path, trajectory_path,
                        hatch_mask_path, skip_seg, skip_agents, api_key, base_url)

                    result_a, result_b = evaluate_single(pred_path, gt_path)
                    r = result_a
                    all_metrics.append({
                        "rta": r.room_type_accuracy,
                        "fna": r.furniture_naming_accuracy,
                        "cls": r.cls,
                    })

                # per-seed retention
                if cls0 is not None and cls0 > 0:
                    per_seed_retentions = [m["cls"] / cls0 for m in all_metrics]
                else:
                    per_seed_retentions = [0.0] * len(all_metrics)

                results[level] = {
                    "multi_seed": True,
                    "sigma_global": sigma_global,
                    "seeds": list(MULTI_SEEDS),
                    "cls_mean": _mean_std([m["cls"] for m in all_metrics]),
                    "rta_mean": _mean_std([m["rta"] for m in all_metrics]),
                    "fna_mean": _mean_std([m["fna"] for m in all_metrics]),
                    "retention_mean": _mean_std(per_seed_retentions),
                    "all_metrics": all_metrics,
                }

        except Exception as e:
            print(f"  [ERROR] σ_g={sigma_global:.2f}dm: {e}")
            import traceback
            traceback.print_exc()
            results[level] = None

    return results


# ============================================================
# B2: 行为轨迹稀疏度 (v2: temporal + multi-seed)
# ============================================================

MULTI_SEEDS = [42, 123, 456, 789, 1024]  # 5 seeds for B2


def run_b2_trajectory_sparsity(
    code: str, image_path: str, smart_device_path: str,
    trajectory_path: str, hatch_mask_path: str,
    output_dir: str, gt_path: str, skip_seg: bool,
    skip_agents: bool = False,
    api_key: str = None, base_url: str = None,
) -> Dict:
    """B2 v2: 轨迹稀疏度 — temporal downsampling + day-level truncation + multi-seed"""
    print(f"\n{'#'*60}")
    print(f"#  B2: 行为轨迹稀疏度 — Temporal Sparsity (multi-seed)")
    print(f"#  temporal downsampling + day-level truncation over {len(MULTI_SEEDS)} seeds")
    print(f"{'#'*60}")

    temp_dir = os.path.join(output_dir, '_temp')
    os.makedirs(temp_dir, exist_ok=True)

    # 定义稀疏度变体: (标签, 采样步长)
    sparsity_variants = [
        ("1天 (密集)", 1),       # step=1 → 100% = full
        ("1/2 采样率", 2),       # step=2 → 50%
        ("1/4 采样率", 4),       # step=4 → 25%
        ("1/7 采样率", 7),       # step=7 → ~14%
    ]

    results = {}
    for label, step in sparsity_variants:
        variant_name = f"B2_Temporal_step{step}"
        try:
            if step <= 1:
                # 全量数据 → 复用已有结果
                existing_final = os.path.join(
                    PROJECT_ROOT, 'output', code, 'yaml', f'03_final_{code}.yaml')
                if os.path.exists(existing_final):
                    print(f"  [复用] {label} 已有结果: {existing_final}")
                    pred_path = existing_final
                    result_a, result_b = evaluate_single(pred_path, gt_path)
                    results[label] = {
                        "result_a": result_a,
                        "result_b": result_b,
                        "step": step,
                        "pred_path": pred_path,
                        "multi_seed": False,
                    }
                    continue
                else:
                    pred_path = run_pipeline_with_inputs(
                        code, variant_name, output_dir,
                        image_path, smart_device_path, trajectory_path,
                        hatch_mask_path, skip_seg, skip_agents, api_key, base_url)
                    result_a, result_b = evaluate_single(pred_path, gt_path)
                    results[label] = {
                        "result_a": result_a,
                        "result_b": result_b,
                        "step": step,
                        "pred_path": pred_path,
                        "multi_seed": False,
                    }
                    continue

            # 多 seed 运行
            all_metrics = []
            for seed in MULTI_SEEDS:
                sparse_traj_path = os.path.join(
                    temp_dir, f'trajectory_{code}_step{step}_seed{seed}.json')
                temporal_downsample_trajectory(trajectory_path, sparse_traj_path, step, seed)

                pred_path = run_pipeline_with_inputs(
                    code, f"{variant_name}_s{seed}", output_dir,
                    image_path, smart_device_path, sparse_traj_path,
                    hatch_mask_path, skip_seg, skip_agents, api_key, base_url)

                result_a, result_b = evaluate_single(pred_path, gt_path)
                r = result_a
                all_metrics.append({
                    "rta": r.room_type_accuracy,
                    "f1": r.furniture_f1,
                    "fna": r.furniture_naming_accuracy,
                    "cls": r.cls,
                    "precision": r.furniture_precision,
                    "recall": r.furniture_recall,
                    "centroid_f1": r.centroid_f1,
                    "centroid_recall": r.centroid_recall,
                })

            # 计算 mean ± std
            def _mean_std(values):
                arr = np.array(values)
                return float(np.mean(arr)), float(np.std(arr, ddof=1))

            results[label] = {
                "step": step,
                "multi_seed": True,
                "seeds": MULTI_SEEDS,
                "cls_mean": _mean_std([m["cls"] for m in all_metrics]),
                "rta_mean": _mean_std([m["rta"] for m in all_metrics]),
                "f1_mean": _mean_std([m["f1"] for m in all_metrics]),
                "fna_mean": _mean_std([m["fna"] for m in all_metrics]),
                "cf1_mean": _mean_std([m["centroid_f1"] for m in all_metrics]),
                "cr_mean": _mean_std([m["centroid_recall"] for m in all_metrics]),
                "all_metrics": all_metrics,
            }

        except Exception as e:
            print(f"  [ERROR] {label}: {e}")
            import traceback
            traceback.print_exc()
            results[label] = None

    return results


# ============================================================
# B3: 图像退化 (v2: realistic corruptions)
# ============================================================

def run_b3_image_degradation(
    code: str, image_path: str, smart_device_path: str,
    trajectory_path: str, hatch_mask_path: str,
    output_dir: str, gt_path: str, skip_seg: bool,
    skip_agents: bool = False,
    api_key: str = None, base_url: str = None,
) -> Dict:
    """B3: 图像退化 — 扫地机器人APP截图真实退化"""
    print(f"\n{'#'*60}")
    print(f"#  B3: 图像退化 — Robot Vacuum APP Screenshot Degradation")
    print(f"#  Rescale / Map Incomplete / SLAM Wall Distortion")
    print(f"{'#'*60}")

    temp_dir = os.path.join(output_dir, '_temp')
    os.makedirs(temp_dir, exist_ok=True)

    degradation_variants = [
        ("Clean (No Degradation)", "clean", {}),
        ("Rescale 0.5x", "rescale", {"scale": 0.5}),
        ("Rescale 0.25x", "rescale", {"scale": 0.25}),
        ("Map Crop 20%", "crop", {"crop_fraction": 0.2}),
        ("SLAM Wall Noise (sigma=1.5)", "wall_noise", {"sigma": 1.5}),
        ("SLAM Wall Noise (sigma=3.0)", "wall_noise", {"sigma": 3.0}),
    ]

    results = {}
    for variant_name, deg_type, config in degradation_variants:
        safe_name = variant_name.replace(' ', '_').replace('%', 'pct').replace('(', '').replace(')', '').replace('=', '').replace('.', '_')
        variant_label = f"B3_{safe_name}"
        try:
            if deg_type == "clean":
                existing_final = os.path.join(
                    PROJECT_ROOT, 'output', code, 'yaml', f'03_final_{code}.yaml')
                if os.path.exists(existing_final):
                    print(f"  [复用] Clean 已有结果: {existing_final}")
                    pred_path = existing_final
                else:
                    pred_path = run_pipeline_with_inputs(
                        code, variant_label, output_dir,
                        image_path, smart_device_path, trajectory_path,
                        hatch_mask_path, skip_seg, skip_agents, api_key, base_url)

            elif deg_type == "rescale":
                degraded_path = os.path.join(temp_dir, f'room_{code}_rescale_{config["scale"]}.png')
                rescale_image(image_path, degraded_path, config["scale"])
                pred_path = run_pipeline_with_inputs(
                    code, variant_label, output_dir,
                    degraded_path, smart_device_path, trajectory_path,
                    hatch_mask_path, False, skip_agents, api_key, base_url)

            elif deg_type == "crop":
                degraded_path = os.path.join(temp_dir, f'room_{code}_crop_{config["crop_fraction"]}.png')
                crop_image(image_path, degraded_path, config["crop_fraction"])
                pred_path = run_pipeline_with_inputs(
                    code, variant_label, output_dir,
                    degraded_path, smart_device_path, trajectory_path,
                    hatch_mask_path, False, skip_agents, api_key, base_url)

            elif deg_type == "wall_noise":
                degraded_path = os.path.join(temp_dir, f'room_{code}_wallnoise_s{config["sigma"]}.png')
                add_wall_noise(image_path, degraded_path, config["sigma"])
                pred_path = run_pipeline_with_inputs(
                    code, variant_label, output_dir,
                    degraded_path, smart_device_path, trajectory_path,
                    hatch_mask_path, False, skip_agents, api_key, base_url)

            result_a, result_b = evaluate_single(pred_path, gt_path)
            results[variant_name] = {
                "result_a": result_a,
                "result_b": result_b,
                "config": config,
                "deg_type": deg_type,
                "pred_path": pred_path,
            }
        except Exception as e:
            print(f"  [ERROR] {variant_name}: {e}")
            import traceback
            traceback.print_exc()
            results[variant_name] = None

    return results


# ============================================================
# 表格输出
# ============================================================

# _safe_val/_safe_std 中短名 → EvaluationResult 属性名映射
_ATTR_MAP = {
    'cls': 'cls',
    'rta': 'room_type_accuracy',
    'f1': 'furniture_f1',
    'fna': 'furniture_naming_accuracy',
    'cf1': 'centroid_f1',
    'cr': 'centroid_recall',
    'precision': 'furniture_precision',
    'recall': 'furniture_recall',
}


def _safe_val(result, attr, default=0.0):
    if result is None:
        return default
    # multi-seed: result 可能包含 mean/std 直接字段
    key = f"{attr}_mean"
    if isinstance(result, dict) and key in result:
        return result[key][0]  # mean
    r = result.get("result_a")
    if r is None:
        return default
    full_attr = _ATTR_MAP.get(attr, attr)
    val = getattr(r, full_attr, None)
    return val if val is not None else default


def _safe_std(result, attr, default=0.0):
    """获取 multi-seed std"""
    if result is None:
        return default
    key = f"{attr}_mean"
    if isinstance(result, dict) and key in result:
        return result[key][1]  # std
    return default


def _sig_stars(p: float) -> str:
    """将 p-value 转为 'p=0.043 *' 格式"""
    if p < 0.001:
        return f"p={p:.3f} ***"
    elif p < 0.01:
        return f"p={p:.3f} **"
    elif p < 0.05:
        return f"p={p:.3f} *"
    else:
        return f"p={p:.3f} n.s."


def print_b1_table(results: Dict):
    """B1 v2: 设备坐标扰动表格 (multi-seed mean±std, per-seed CLS Retention)"""
    lines = []
    lines.append(f"\n{'='*105}")
    lines.append(f"  B1: 设备坐标扰动 — Correlated Perturbation (multi-seed)")
    lines.append(f"  σ_global = noise_dm/3, σ_local = noise_dm/9, seeds={MULTI_SEEDS}")
    lines.append(f"{'='*105}")
    header = (f"  {'σ_global (dm)':<14} {'CLS':>14} {'RTA':>14} {'FNA':>14} "
              f"{'CLS Retention':>14}")
    sep = f"  {'-'*94}"
    lines.append(header)
    lines.append(sep)

    for level in [0, 2, 5, 10, 15, 20]:
        r = results.get(level)
        if r is None:
            lines.append(f"  {'N/A':<14}")
            continue

        sigma_g = level / 3.0
        multi = r.get("multi_seed", False)

        if multi:
            cls_s = f"{_safe_val(r, 'cls'):.1%} ±{_safe_std(r, 'cls'):.1%}"
            rta_s = f"{_safe_val(r, 'rta'):.1%} ±{_safe_std(r, 'rta'):.1%}"
            fna_s = f"{_safe_val(r, 'fna'):.1%} ±{_safe_std(r, 'fna'):.1%}"
            ret_mean, ret_std = r.get("retention_mean", (0.0, 0.0))
            retention_s = f"{ret_mean:.1%} ±{ret_std:.1%}"
        else:
            cls_s = f"{_safe_val(r, 'cls'):.1%}"
            rta_s = f"{_safe_val(r, 'rta'):.1%}"
            fna_s = f"{_safe_val(r, 'fna'):.1%}"
            retention_s = f"{1.0:.1%}"

        lines.append(
            f"  {sigma_g:<14.2f} {cls_s:>14} {rta_s:>14} "
            f"{fna_s:>14} {retention_s:>14}"
        )

    lines.append(sep)
    lines.append("  Seeds are repeated perturbations of one home, not independent statistical units; no p-value is reported.")
    lines.append("  For inference, average seeds within each home and run paired tests/bootstrap CIs across homes.")

    # σ_g=3.33dm (对应 noise_dm=10) 处评估
    r10 = results.get(10)
    r0 = results.get(0)
    if r0 and r10:
        if r10.get("multi_seed") and "retention_mean" in r10:
            retention_10 = r10["retention_mean"][0]
        else:
            retention_10 = _safe_val(r10, 'cls') / (_safe_val(r0, 'cls') or 1.0)
        lines.append(f"  CLS Retention @σ_g=3.33dm: {retention_10:.1%}")
        if retention_10 >= 0.90:
            lines.append(f"  [OK] σ_g=3.33dm 内 CLS 保持率 >= 90%")
        else:
            lines.append(f"  [WARN] σ_g=3.33dm 内 CLS 保持率 {retention_10:.1%} (< 90%)")

    return lines


def print_b2_table(results: Dict):
    """B2 v2: 轨迹稀疏度表格 (含 multi-seed mean ± std)"""
    lines = []
    lines.append(f"\n{'='*100}")
    lines.append(f"  B2: 行为轨迹稀疏度 — Temporal Downsampling (multi-seed)")
    lines.append(f"  Seeds: {MULTI_SEEDS}")
    lines.append(f"{'='*100}")
    header = (f"  {'稀疏度':<16} {'Step':>5} {'CLS':>16} {'RTA':>16} "
              f"{'F1':>16} {'CentroidF1':>16}")
    sep = f"  {'-'*81}"
    lines.append(header)
    lines.append(sep)

    for label, r in results.items():
        if r is None:
            lines.append(f"  {label:<16} {'ERROR':>5}")
            continue
        step = r.get("step", 1)
        multi = r.get("multi_seed", False)

        if multi:
            cls_s = f"{_safe_val(r, 'cls'):.1%} ±{_safe_std(r, 'cls'):.1%}"
            rta_s = f"{_safe_val(r, 'rta'):.1%} ±{_safe_std(r, 'rta'):.1%}"
            f1_s = f"{_safe_val(r, 'f1'):.1%} ±{_safe_std(r, 'f1'):.1%}"
            cf1_s = f"{_safe_val(r, 'cf1'):.1%} ±{_safe_std(r, 'cf1'):.1%}"
        else:
            cls_s = f"{_safe_val(r, 'cls'):.1%}"
            rta_s = f"{_safe_val(r, 'rta'):.1%}"
            f1_s = f"{_safe_val(r, 'f1'):.1%}"
            cf1_s = f"{_safe_val(r, 'cf1'):.1%}"

        lines.append(f"  {label:<16} {step:>5} {cls_s:>16} {rta_s:>16} "
                     f"{f1_s:>16} {cf1_s:>16}")

    lines.append(sep)
    return lines


def print_b3_table(results: Dict):
    """B3: 图像退化表格 (APP截图真实退化)"""
    lines = []
    lines.append(f"\n{'='*100}")
    lines.append(f"  B3: 图像退化 — Robot Vacuum APP Screenshot Degradation")
    lines.append(f"  Rescale (低分辨率) / Map Crop (地图缺角) / SLAM Wall Noise (建图畸变)")
    lines.append(f"  Note: 图像级退化需要重新分割 (skip_seg=False)")
    lines.append(f"{'='*100}")
    header = (f"  {'退化类型':<32} {'F1':>8} {'FNA':>8} {'CLS':>8} "
              f"{'小家具Recall':>12} {'CentroidR':>10}")
    sep = f"  {'-'*82}"
    lines.append(header)
    lines.append(sep)

    clean_result = results.get("Clean (No Degradation)")

    for variant_name, r in results.items():
        if r is None:
            lines.append(f"  {variant_name:<32} {'ERROR':>8}")
            continue
        f1_val = _safe_val(r, 'furniture_f1')
        fna_val = _safe_val(r, 'furniture_naming_accuracy')
        cls_val = _safe_val(r, 'cls')
        cr_val = _safe_val(r, 'centroid_recall')

        small_recall = 0.0
        result_eval = r.get("result_a")
        if result_eval:
            pc = result_eval.per_class.get('small (<50)', {})
            if pc.get('gt', 0) > 0:
                small_recall = pc.get('tp', 0) / pc['gt']

        lines.append(
            f"  {variant_name:<32} {f1_val:>7.1%} {fna_val:>7.1%} "
            f"{cls_val:>7.1%} {small_recall:>11.1%} {cr_val:>9.1%}"
        )

    lines.append(sep)

    if clean_result:
        clean_f1 = _safe_val(clean_result, 'furniture_f1')
        lines.append(f"  --- F1 drop vs Clean ---")
        for variant_name, r in results.items():
            if r and variant_name != "Clean (No Degradation)":
                f1_val = _safe_val(r, 'furniture_f1')
                drop = clean_f1 - f1_val
                lines.append(f"  {variant_name:<32} Delta F1: {drop:+.1%}")
        lines.append(f"")
        lines.append(f"  [Interpretation]")
        lines.append(f"  - Rescale: tests robustness to low-resolution screenshots")
        lines.append(f"  - Map Crop: tests robustness to incomplete floor scans")
        lines.append(f"  - SLAM Noise: tests robustness to mapping distortions")

    return lines


# ============================================================
# 保存为结构化数据
# ============================================================

def save_structured_results(all_results: dict, output_dir: str, code: str):
    """将结构化结果保存为 JSON 以供画图脚本读取"""
    structured = {}
    for group_key, group_data in all_results.items():
        structured[group_key] = {}
        for variant_key, r in group_data.items():
            if r is None:
                structured[group_key][variant_key] = None
                continue

            entry = {}

            # multi-seed (B1 & B2): 直接使用 mean 值
            if r.get("multi_seed"):
                for attr in ["cls", "rta", "f1", "fna", "cf1", "cr"]:
                    key = f"{attr}_mean"
                    if key in r:
                        entry[attr] = r[key][0]
                        entry[f"{attr}_std"] = r[key][1]
                # per-seed retention (B1 only)
                if "retention_mean" in r:
                    entry["retention"] = r["retention_mean"][0]
                    entry["retention_std"] = r["retention_mean"][1]
                entry["multi_seed"] = True
                entry["step"] = r.get("step")
                entry["sigma_global"] = r.get("sigma_global")
                entry["seeds"] = r.get("seeds")
                entry["all_metrics"] = r.get("all_metrics")
            else:
                result_eval = r.get("result_a")
                if result_eval is None:
                    structured[group_key][variant_key] = None
                    continue
                entry = {
                    "rta": result_eval.room_type_accuracy,
                    "f1": result_eval.furniture_f1,
                    "fna": result_eval.furniture_naming_accuracy,
                    "cls": result_eval.cls,
                    "precision": result_eval.furniture_precision,
                    "recall": result_eval.furniture_recall,
                    "centroid_f1": result_eval.centroid_f1,
                    "centroid_recall": result_eval.centroid_recall,
                }
                per_class = {}
                for cls_name, pc in result_eval.per_class.items():
                    recall = pc['tp'] / pc['gt'] if pc['gt'] > 0 else 0
                    per_class[cls_name] = {
                        'gt': pc['gt'], 'tp': pc['tp'], 'recall': recall,
                        'tp_correct_name': pc.get('tp_correct_name', 0),
                    }
                entry['per_class'] = per_class

            if 'level' in r:
                entry['level'] = r['level']
            if 'step' in r:
                entry['step'] = r['step']
            if 'deg_type' in r:
                entry['deg_type'] = r['deg_type']

            structured[group_key][variant_key] = entry

    json_path = os.path.join(output_dir, f'noise_robustness_{code}.json')
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(structured, f, indent=2, ensure_ascii=False)
    print(f"\n结构化结果已保存至: {json_path}")
    return json_path


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='实验组 B v2：噪声鲁棒性 — correlated perturbation + temporal sparsity + realistic corruption')
    parser.add_argument('--code', type=str, default='0622',
                        help='家庭编码 (默认: 0622)')
    parser.add_argument('--group', type=str, action='append', default=None,
                        choices=['b1', 'b2', 'b3', 'all'],
                        help='执行指定实验组 (可多次指定): b1, b2, b3, all')
    parser.add_argument('--skip-seg', action='store_true',
                        help='跳过房间分割')
    parser.add_argument('--skip-agents', action='store_true',
                        help='跳过所有 LLM Agents (仅运行 CV 管线)')
    parser.add_argument('--image-dir', type=str, default='input/photo',
                        help='户型图目录')
    parser.add_argument('--traj-dir', type=str, default='input/traj',
                        help='轨迹数据目录')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='输出目录')

    args = parser.parse_args()

    code = args.code

    if args.group is None or 'all' in args.group:
        groups = ['b1', 'b2', 'b3']
    else:
        groups = args.group

    api_key = os.environ.get("FURNITURE_API_KEY")
    base_url = os.environ.get("FURNITURE_BASE_URL")
    if not api_key:
        print("[错误] 环境变量 FURNITURE_API_KEY 未设置")
        sys.exit(1)

    # 路径
    gt_path = os.path.join(PROJECT_ROOT, 'GT', f'layout_{code}.yaml')
    if not os.path.exists(gt_path):
        print(f"[错误] GT 文件不存在: {gt_path}")
        sys.exit(1)

    image_path = os.path.join(PROJECT_ROOT, args.image_dir, f'room_{code}.png')
    if not os.path.exists(image_path):
        print(f"[错误] 图片不存在: {image_path}")
        sys.exit(1)

    smart_device_path = os.path.join(PROJECT_ROOT, 'input', 'Smart_device',
                                     f'smartDevice_{code}.yaml')
    if not os.path.exists(smart_device_path):
        print(f"[错误] 智能设备文件不存在: {smart_device_path}")
        sys.exit(1)

    traj_dir = os.path.join(PROJECT_ROOT, args.traj_dir, code)
    trajectory_path = None
    if os.path.isdir(traj_dir):
        jsons = sorted(Path(traj_dir).glob('*.json'),
                       key=lambda p: p.stat().st_size, reverse=True)
        if jsons:
            trajectory_path = str(jsons[0])
    if not trajectory_path:
        print(f"[错误] 轨迹文件不存在: {traj_dir}")
        sys.exit(1)

    hatch_mask_path = os.path.join(PROJECT_ROOT, 'output', code, 'masks', 'hatch_mask.png')
    if not os.path.exists(hatch_mask_path):
        alt_hm = os.path.join(PROJECT_ROOT, 'photo2yaml', 'output', 'masks', 'hatch_mask.png')
        if os.path.exists(alt_hm):
            hatch_mask_path = alt_hm
        else:
            print(f"[警告] hatch_mask.png 不存在，家具检测可能缺少红框信息")
            hatch_mask_path = None

    output_dir = os.path.join(PROJECT_ROOT, args.output or f'output/{code}/noise_robustness')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  实验组 B v2：噪声鲁棒性 (Review 改进版)")
    print("=" * 60)
    print(f"  家庭编码:          {code}")
    print(f"  户型图:            {image_path}")
    print(f"  智能设备:          {smart_device_path}")
    print(f"  轨迹数据:          {trajectory_path}")
    print(f"  Hatch Mask:        {hatch_mask_path}")
    print(f"  输出目录:          {output_dir}")
    print(f"  实验组:            {groups}")
    print(f"  跳过分割:          {args.skip_seg}")
    print(f"  跳过 LLM:          {args.skip_agents}")
    print("=" * 60)

    all_results = {}
    all_lines = []

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    lines_header = [
        f"实验组 B v2：噪声鲁棒性 (Review 改进版) — {code}",
        f"运行时间: {datetime.now().isoformat()}",
        f"",
        f"改进要点:",
        f"  B1: correlated perturbation + multi-seed mean±std (σ_global=noise/3) + Retention 指标",
        f"  B2: temporal downsampling + multi-seed mean±std",
        f"  B3: realistic corruption (JPEG / Gamma / Noise / Blur)",
        f"",
    ]
    all_lines.extend(lines_header)

    # ── B1 ──
    if 'b1' in groups:
        b1_results = run_b1_device_perturbation(
            code, image_path, smart_device_path, trajectory_path,
            hatch_mask_path, output_dir, gt_path, args.skip_seg,
            skip_agents=args.skip_agents, api_key=api_key, base_url=base_url)
        for level in [0, 2, 5, 10, 15, 20]:
            if b1_results.get(level):
                b1_results[level]['level'] = level
        all_results['b1'] = b1_results
        b1_lines = print_b1_table(b1_results)
        for line in b1_lines:
            print(line)
        all_lines.extend(b1_lines)

    # ── B2 ──
    if 'b2' in groups:
        b2_results = run_b2_trajectory_sparsity(
            code, image_path, smart_device_path, trajectory_path,
            hatch_mask_path, output_dir, gt_path, args.skip_seg,
            skip_agents=args.skip_agents, api_key=api_key, base_url=base_url)
        all_results['b2'] = b2_results
        b2_lines = print_b2_table(b2_results)
        for line in b2_lines:
            print(line)
        all_lines.extend(b2_lines)

    # ── B3 ──
    if 'b3' in groups:
        b3_results = run_b3_image_degradation(
            code, image_path, smart_device_path, trajectory_path,
            hatch_mask_path, output_dir, gt_path, args.skip_seg,
            skip_agents=args.skip_agents, api_key=api_key, base_url=base_url)
        all_results['b3'] = b3_results
        b3_lines = print_b3_table(b3_results)
        for line in b3_lines:
            print(line)
        all_lines.extend(b3_lines)

    # ── 保存结果 ──
    eval_dir = os.path.join(PROJECT_ROOT, 'output', code, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    txt_path = os.path.join(eval_dir, f'noise_robustness_{timestamp}.txt')
    all_lines.append(f"\n  CLS = 0.45 x FNA + 0.35 x F1_IoU + 0.20 x RTA")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_lines))
    print(f"\n文本结果已保存至: {txt_path}")

    json_path = save_structured_results(all_results, eval_dir, code)
    print(f"\n✅ 实验组 B v2 全部完成！")
    print(f"   文本结果: {txt_path}")
    print(f"   结构化数据: {json_path}")


if __name__ == '__main__':
    main()
