"""
消融实验主脚本 — ablation_experiments.py

论文级四组消融实验。默认验证四个概念贡献，且每个变体只改变所属实验组
定义的输入或模块边界：
  1. 模态贡献：Layout / Smart Device / Human Trajectory
  2. CV-LLM 混合检测：受限修正 / 纯 CV / Free-form / Direct LLM
  3. 双源家具召回：Hatch / Trajectory Loop
  4. 推理策略：7 层房间优先级、行为先验、命名词表、形状约束

输出 Global 和 Room-aligned 两套协议下的 RoomF1 / LocF1 / SemF1 / FNA /
SemFDR，并额外报告不依赖 GT 的 Protocol Violation Rate。

用法:
  python baselines/ablation_experiments.py --code 0622

  # 跳过房间分割
  python baselines/ablation_experiments.py --code 0622 --skip-seg

  # 批量
  python baselines/ablation_experiments.py --codes 0622,0701
"""
import os, sys, argparse, yaml, json, shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── 加载 .env (与 main.py 一致，自动检测编码 UTF-8 / UTF-16) ──
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
        print("[警告] 无法解码 .env 文件，尝试以 UTF-8 读取（忽略错误）")
        load_dotenv(dotenv_path=_env_path, encoding='utf-8', errors='ignore')
    print(f"[dotenv] 已加载: {_env_path}")

from pipeline import Pipeline
from evaluation import evaluate_single
from config import load_furniture_allowed, get_smart_device_names


# Protocol-level constraint metric. This is separate from end-to-end
# semantic metrics and does not use ground truth.
FURNITURE_ALLOWED = load_furniture_allowed()
SMART_DEVICE_NAMES = get_smart_device_names()


def _normalize_fur_name(name: str) -> str:
    mapping = {
        'coffee table': 'coffee_table', 'coffee_table': 'coffee_table',
        'dining table': 'dining_table', 'dining_table': 'dining_table',
        'shoe cabinet': 'shoe_cabinet', 'shoe_cabinet': 'shoe_cabinet',
        'coat rack': 'coat_rack', 'coat_rack': 'coat_rack',
        'washing machine': 'washing_machine',
        'tv cabinet': 'tv_cabinet', 'tv stand': 'tv_stand',
        'night stand': 'nightstand', 'book shelf': 'bookshelf',
    }
    key = (name or '').lower().replace('_', ' ').strip()
    return mapping.get(key, key.replace(' ', '_'))


def _compute_protocol_violation_rate(
    naming_records: list,
    room_type_by_id: dict,
) -> tuple:
    """Return (rate, violations, non-device predictions) for naming records."""
    smart_names = {_normalize_fur_name(name) for name in SMART_DEVICE_NAMES}
    violations = total = 0
    for naming in naming_records:
        if isinstance(naming, dict):
            name = naming.get('name', '')
            room_id = naming.get('room_id', '')
        else:
            name = getattr(naming, 'name', '')
            room_id = getattr(naming, 'room_id', '')

        room_type = (room_type_by_id.get(room_id, 'other') or 'other').lower()
        allowed = FURNITURE_ALLOWED.get(
            room_type, FURNITURE_ALLOWED.get('other', []))
        allowed_names = {_normalize_fur_name(name) for name in allowed}
        normalized = _normalize_fur_name(name)
        if not normalized or normalized in smart_names:
            continue
        total += 1
        if normalized not in allowed_names:
            violations += 1

    return (violations / total if total else 0.0), violations, total


def compute_pre_final_protocol_violation_rate(
    furniture_namings: list,
    room_analyses: list,
) -> tuple:
    """Measure ProtV immediately after naming and before final-output repair.

    The snapshot contains LLM/protected-device names before vocabulary repair,
    duplicate-name resolution, geometric post-processing, final YAML merging,
    and overlap correction.  It is therefore a mechanism diagnostic, not a
    final-output conformance check.
    """
    room_type_by_id = {
        (room.get('room_id') if isinstance(room, dict) else room.room_id):
        (room.get('room_type') if isinstance(room, dict) else room.room_type)
        for room in room_analyses
    }
    return _compute_protocol_violation_rate(furniture_namings, room_type_by_id)


def compute_protocol_violation_rate(pred_yaml_path: str) -> tuple:
    """Return the final-YAML rate for backward-compatible ad-hoc inspection.

    Ablation tables use :func:`compute_pre_final_protocol_violation_rate`
    instead, so their ProtV column is measured before final-output repair.
    """
    if not os.path.exists(pred_yaml_path):
        return 0.0, 0, 0

    with open(pred_yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    naming_records = []
    room_type_by_id = {}
    for index, room in enumerate(data.get('house', {}).get('rooms', [])):
        room_id = room.get('id', room.get('name', f'room_{index}'))
        room_type_by_id[room_id] = room.get('name', '')
        for furniture in room.get('furniture', []):
            naming_records.append({
                'name': furniture.get('name', ''),
                'room_id': room_id,
            })

    return _compute_protocol_violation_rate(naming_records, room_type_by_id)


def compute_hallucination_rate(pred_yaml_path: str, gt_path: str = None) -> tuple:
    """Backward-compatible alias; the GT path is intentionally ignored."""
    return compute_protocol_violation_rate(pred_yaml_path)

# ============================================================
# 消融实验变体定义
# ============================================================

# 1) 模态贡献：这是“输入是否存在”的全链路消融。skip_trajectory 会同时
# 从家具检测、RoomAgent 和 BehaviorAgent 移除人的轨迹，不能与
# skip_trajectory_loop（仅去掉一个 CV 召回源）混用。
MODAL_VARIANTS = {
    "Layout Only": {"skip_trajectory": True, "skip_smart_device": True},
    "+ Device": {"skip_trajectory": True},
    "+ Trajectory": {"skip_smart_device": True},
    "Full (All Modalities)": {},
}

# 2) CV-LLM 混合检测：固定三类输入，只改变家具检测的决策方式。
# Free-form 仍使用 CV 候选作为输入，但允许 LLM add/delete/merge/adjust；
# Direct LLM Generation 完全不把 CV 候选交给 LLM。两者只改变家具检测
# 决策边界，命名漏斗保持与 Ours 相同，以免混淆检测与命名贡献。
HYBRID_VARIANTS = {
    "CV-only": {"skip_llm_correction": True},
    "Free-form LLM": {"use_free_form_llm": True},
    "Direct LLM Generation": {"use_direct_llm_gen": True},
    "Constrained (Ours)": {},
}
# Backward-compatible name used by older notebooks and result parsers.
LLM_VARIANTS = HYBRID_VARIANTS

# 3) 双源家具召回：只关闭一个 CV 候选源；轨迹输入本身仍传给下游。
DETECTION_VARIANTS = {
    "Full (Dual-source)": {},
    "No Hatch": {"skip_hatch": True},
    "No Trajectory Loop": {"skip_trajectory_loop": True},
}

# 4) 推理策略：每次只去掉一个确定性推理约束。
REASONING_VARIANTS = {
    "Full (Reasoning)": {},
    "No 7-Layer Priority": {"skip_layered_priority": True},
    "No Behavior Prior": {"skip_behavior_prior": True},
    "No Allowed List": {"skip_allowed_list": True},
    "No Shape Constraint": {"skip_shape_constraint": True},
}

# 兼容早期“专利表1”调用方。该配置保留为补充实验，不放进默认四组，
# 从而避免把“轨迹输入消融”和“轨迹闭环候选消融”混为同一个结论。
PIPELINE_VARIANTS = {
    "Full (Ours)": {},
    "No Trajectory-Loop Recall": {"skip_trajectory_loop": True},
    "No LLM Geometry Correction": {"skip_llm_correction": True},
    "No 7-Layer Room Priority": {"skip_layered_priority": True},
    "No Behavior Agent": {"skip_behavior_module": True},
    "No Naming Funnel": {"skip_naming_funnel": True},
}

# 5) Core-module ablation used to test whether each claimed module contributes
# to the final reconstruction.  The trajectory-loop recall and LLM geometry
# correction are evaluated as one coupled detection/geometry variant because
# the two stages jointly determine the final furniture boxes.
CORE_MODULE_VARIANTS = {
    "Full (Ours)": {},
    "No Room Semantic Module": {"skip_room_module": True},
    "No Behavior Semantic Module": {"skip_behavior_module": True},
    "No Furniture Naming Module": {"skip_naming_module": True},
    "No Furniture Geometry Correction": {
        "skip_trajectory_loop": True,
        "skip_llm_correction": True,
    },
    "No 7-Layer Room Priority": {"skip_layered_priority": True},
}


def _validate_core_module_variants(variants: dict) -> None:
    """Fail fast if a core ablation disables an undeclared module combination."""
    if variants.get("Full (Ours)") != {}:
        raise ValueError("Core ablation baseline must use an empty configuration")
    for name, config in variants.items():
        if name == "Full (Ours)":
            continue
        if not all(value is True for value in config.values()):
            raise ValueError(
                f"{name} must use boolean ablation flags, got {config!r}")
        flags = set(config)
        if len(flags) == 1:
            continue
        if flags == {"skip_trajectory_loop", "skip_llm_correction"}:
            continue
        raise ValueError(
            f"{name} must remove one core module or the declared trajectory/geometry pair, "
            f"got {config!r}")


_validate_core_module_variants(CORE_MODULE_VARIANTS)

# ``all`` remains the historical four-group selection for reproducibility.
# The default command is intentionally the focused core-module experiment.
PRIMARY_GROUP_KEYS = ("modal", "hybrid", "detection", "reasoning")

ABLATION_GROUPS = {
    "core": {
        "name": "Core Module Removal",
        "variants": CORE_MODULE_VARIANTS,
        "table_type": "core",
        "description": "Full system versus single-module removals plus the coupled trajectory/geometry removal",
    },
    "modal": {
        "name": "Modal Contribution",
        "variants": MODAL_VARIANTS,
        "table_type": "modal",
        "description": "累加式验证 Layout、Smart Device 与 Human Trajectory 的独立及互补贡献",
    },
    "hybrid": {
        "name": "Hybrid CV-LLM Detection",
        "variants": HYBRID_VARIANTS,
        "table_type": "hybrid",
        "description": "固定输入，比较受限动作修正、纯 CV、Free-form 和 Direct LLM",
    },
    "detection": {
        "name": "Dual-source Furniture Detection",
        "variants": DETECTION_VARIANTS,
        "table_type": "detection",
        "description": "只移除 Hatch 或 Trajectory-loop 一个家具候选源",
    },
    "reasoning": {
        "name": "Reasoning Strategy",
        "variants": REASONING_VARIANTS,
        "table_type": "reasoning",
        "description": "逐一移除房间或命名阶段的单个推理约束",
    },
}


# ============================================================
# 辅助函数
# ============================================================

def _ordered_variant_names(variants: dict) -> list:
    """返回固定的消融执行/展示顺序：Full 基准优先，其后为其余变体。"""
    baseline = next(
        (name for name in variants
         if name == "Full (Ours)"
         or name.startswith("Full (")
         or name == "Constrained (Ours)"),
        None,
    )
    if baseline is not None:
        return [baseline, *(name for name in variants if name != baseline)]
    return [
        *(["Full (Ours)"] if "Full (Ours)" in variants else []),
        *(name for name in variants if name != "Full (Ours)"),
    ]


def _resolve_group_infos(requested_groups):
    """Resolve CLI groups without duplicate execution.

    With no explicit group, run the focused core-module experiment.  ``all``
    keeps the historical four primary paper groups.  The legacy ``module``
    group is supplementary and is only added when explicitly requested,
    including for ``--group all --group module``.
    """
    requested_groups = list(requested_groups or [])
    if not requested_groups:
        selected = ["core"]
        include_all = False
    else:
        include_all = "all" in requested_groups
        selected = list(PRIMARY_GROUP_KEYS) if include_all else []
    if not include_all:
        for key in requested_groups:
            if key in ABLATION_GROUPS and key not in selected:
                selected.append(key)
    infos = []
    if "module" in requested_groups:
        infos.append(("module", {
            "name": "Pipeline Component Removal (Supplementary)",
            "variants": PIPELINE_VARIANTS,
            "table_type": "module",
            "description": "兼容补充表：每次移除五步管线中的一个组件",
        }))
    infos.extend((key, ABLATION_GROUPS[key]) for key in selected)
    return infos


def run_pipeline_with_ablation(
    code: str, ablation: dict, variant_name: str,
    image_path: str, smart_device_path: str, trajectory_path: str,
    output_dir: str, skip_seg: bool = False,
    api_key: str = None, base_url: str = None,
    return_protocol_input: bool = False,
) -> object:
    """运行带消融配置的 Pipeline。

    The historical default returns the final YAML path.  The ablation runner
    opts into the pre-final naming snapshot needed for its ProtV measurement.
    """
    print(f"\n{'─'*55}")
    print(f"  Ablation Variant: {variant_name}")

    pipeline = Pipeline(api_key=api_key, base_url=base_url, ablation=ablation, code=code)

    safe_name = variant_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    variant_output_dir = os.path.join(output_dir, f"ablation_{code}_{safe_name}")
    os.makedirs(variant_output_dir, exist_ok=True)

    yaml_dir = os.path.join(variant_output_dir, 'yaml')
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
        seg_path = pipeline.run_seg_step(image_path, variant_output_dir)
        src_yaml = os.path.join(variant_output_dir, 'floorplan_real.yaml')
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

    # ── Step 2: 家具检测 + 坐标统一 + 智能设备匹配 ──
    greyroom_path = os.path.join(variant_output_dir, 'masks', 'greyroom.png')
    hatch_mask_path = os.path.join(variant_output_dir, 'masks', 'hatch_mask.png')
    hm_path = hatch_mask_path if os.path.exists(hatch_mask_path) else None
    seg_image = greyroom_path if os.path.exists(greyroom_path) else image_path

    world_yaml_path, trajectory_data = pipeline.step2_furniture_and_devices(
        seg_image, seg_path, trajectory_path,
        smart_device_path, hm_path, variant_output_dir)

    dst_02 = os.path.join(yaml_dir, f'02_furniture_world_{code}.yaml')
    if os.path.exists(world_yaml_path) and world_yaml_path != dst_02:
        shutil.copy2(world_yaml_path, dst_02)
        world_yaml_path = dst_02

    # ── Step 3-5: LLM Agents ──
    room_analyses = pipeline.room_analysis_step(world_yaml_path, trajectory_data)
    behavior_analyses = pipeline.behavior_analysis_step(
        world_yaml_path, trajectory_data, room_analyses)
    final_path = pipeline.step5_furniture_naming_and_merge(
        world_yaml_path, room_analyses, behavior_analyses, variant_output_dir)

    dst_final = os.path.join(yaml_dir, f'03_final_{code}.yaml')
    if os.path.exists(final_path) and final_path != dst_final:
        shutil.copy2(final_path, dst_final)
        final_path = dst_final

    print(f"  Output: {final_path}")
    if return_protocol_input:
        return {
            'final_path': dst_final,
            'pre_final_namings': pipeline.last_pre_final_namings,
            'room_analyses': pipeline.last_pre_final_room_analyses,
        }
    return dst_final


def run_ablation_group(
    code: str, group_key: str, group_info: dict,
    image_path: str, smart_device_path: str, trajectory_path: str,
    output_dir: str, gt_path: str, skip_seg: bool = False,
    api_key: str = None, base_url: str = None,
) -> Dict:
    """运行一组消融实验并评估"""
    variants = group_info["variants"]
    group_name = group_info["name"]

    print(f"\n{'#'*60}")
    print(f"#  {group_name} ({group_key})")
    print(f"{'#'*60}")

    results = {}
    # Full (Ours) 必须先运行并完成评估，再依次执行其余消融变体。
    # 显式构造顺序，避免未来调整字典来源或合并配置时改变基准执行顺序。
    variant_order = _ordered_variant_names(variants)
    for variant_name in variant_order:
        ablation_config = variants[variant_name]
        try:
            # 每个变体都重新运行，避免复用由不同代码、模型或 prompt 版本产生的旧结果。
            pipeline_result = run_pipeline_with_ablation(
                code, ablation_config, variant_name,
                image_path, smart_device_path, trajectory_path,
                output_dir, skip_seg, api_key, base_url,
                return_protocol_input=True,
            )
            pred_path = pipeline_result['final_path']
            result_a, result_b = evaluate_single(pred_path, gt_path)
            protocol_rate, protocol_n, protocol_total = compute_pre_final_protocol_violation_rate(
                pipeline_result['pre_final_namings'],
                pipeline_result['room_analyses'],
            )
            results[variant_name] = {
                "result_a": result_a,
                "result_b": result_b,
                "protocol_rate": protocol_rate,
                "protocol_n": protocol_n,
                "protocol_total": protocol_total,
                "pred_path": pred_path,
            }
        except Exception as e:
            print(f"  [ERROR] {variant_name}: {e}")
            import traceback
            traceback.print_exc()
            results[variant_name] = None

    return results


def _build_module_table(group_name: str, variants: dict, results: dict,
                        description: str = None) -> list:
    """构建一组消融表，并同时列出 Global 与 Room-aligned 协议。

    ``result_a``/``result_b`` 已在独立评估阶段冻结；此处只负责展示，
    不重新匹配、不使用 GT 修正预测结果。保留旧函数名以兼容已有脚本。
    """
    lines = []
    lines.append(f"{'='*95}")
    lines.append(f"  {group_name}")
    if description:
        lines.append(f"  Design: {description}")
    if group_name == "Core Module Removal":
        lines.append("  Each ablation variant removes one core module; the trajectory/geometry variant removes the declared pair.")
    else:
        lines.append("  Variants change only their declared boundary; all other inputs and evaluation settings are frozen.")
    lines.append("  ProtV is measured after naming and before final-output repair; SemFDR is the end-to-end semantic false-discovery rate.")
    lines.append("  dSemF1/dFNA are variant minus the full-system baseline; negative values indicate degradation.")
    lines.append(f"{'='*95}")

    ordered_names = _ordered_variant_names(variants)
    baseline_name = ordered_names[0] if ordered_names else None

    def metric_delta(result, baseline_result, field):
        if result is None or baseline_result is None:
            return None
        value = getattr(result, field, None)
        baseline_value = getattr(baseline_result, field, None)
        if value is None or baseline_value is None:
            return None
        return value - baseline_value

    baseline_a = (results.get(baseline_name, {}) or {}).get("result_a") if baseline_name else None
    baseline_b = (results.get(baseline_name, {}) or {}).get("result_b") if baseline_name else None

    lines.append("  Protocol: A. Global matching")
    header = (f"  {'Variant':<30} {'RoomF1':>8} {'LocF1':>8} {'SemF1':>8} "
              f"{'FNA':>8} {'SemFDR':>8} {'ProtV':>8} {'dSemF1':>8} {'dFNA':>8}")
    sep = f"  {'-'*78}"
    lines.append(header)
    lines.append(sep)

    for variant_name in _ordered_variant_names(variants):
        result = results.get(variant_name)
        if result is None:
            lines.append(f"  {variant_name:<30} {'ERROR':>8}")
            continue
        r = result.get("result_a")
        if r is None or r.furniture_f1 is None:
            lines.append(f"  {variant_name:<30} {'N/A':>8}")
            continue
        sem_fdr = 1.0 - r.semantic_precision if r.semantic_precision is not None else None
        protocol_rate = result.get("protocol_rate")
        d_sem = metric_delta(r, baseline_a, "semantic_f1")
        d_fna = metric_delta(r, baseline_a, "furniture_naming_accuracy")
        lines.append(f"  {variant_name:<30} {r.room_f1:>7.1%} {r.furniture_f1:>7.1%} "
                     f"{r.semantic_f1:>7.1%} {r.furniture_naming_accuracy:>7.1%} "
                     f"{(sem_fdr if sem_fdr is not None else float('nan')):>7.1%} "
                     f"{(protocol_rate if protocol_rate is not None else float('nan')):>7.1%} "
                     f"{(d_sem if d_sem is not None else float('nan')):>+7.1%} "
                     f"{(d_fna if d_fna is not None else float('nan')):>+7.1%}")

    lines.append(sep)
    lines.append("  RoomF1=房间语义F1  LocF1=家具检测F1(IoU>0.2)  SemF1=端到端语义F1")
    lines.append("  FNA=定位条件下命名准确率  SemFDR=1-SemanticPrecision")
    lines.append("")
    lines.append("  Protocol: B. Room-aligned matching")
    lines.append(header)
    lines.append(sep)

    for variant_name in _ordered_variant_names(variants):
        result = results.get(variant_name)
        if result is None:
            lines.append(f"  {variant_name:<30} {'ERROR':>8}")
            continue
        r = result.get("result_b")
        if r is None or r.furniture_f1 is None:
            lines.append(f"  {variant_name:<30} {'N/A':>8}")
            continue
        sem_fdr = 1.0 - r.semantic_precision if r.semantic_precision is not None else None
        protocol_rate = result.get("protocol_rate")
        d_sem = metric_delta(r, baseline_b, "semantic_f1")
        d_fna = metric_delta(r, baseline_b, "furniture_naming_accuracy")
        lines.append(f"  {variant_name:<30} {r.room_f1:>7.1%} {r.furniture_f1:>7.1%} "
                     f"{r.semantic_f1:>7.1%} {r.furniture_naming_accuracy:>7.1%} "
                     f"{(sem_fdr if sem_fdr is not None else float('nan')):>7.1%} "
                     f"{(protocol_rate if protocol_rate is not None else float('nan')):>7.1%} "
                     f"{(d_sem if d_sem is not None else float('nan')):>+7.1%} "
                     f"{(d_fna if d_fna is not None else float('nan')):>+7.1%}")

    lines.append(sep)
    return lines


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='消融实验 — 模态、混合检测、双源召回与推理策略')
    parser.add_argument('--code', type=str, action='append', default=None,
                        help='家庭编码 (如 0622)，可多次指定')
    parser.add_argument('--codes', type=str, default='0622',
                        help='用逗号分隔的编码列表 (如 0622,0701)')
    parser.add_argument('--group', type=str, action='append', default=None,
                        choices=list(ABLATION_GROUPS.keys()) + ['all', 'module'],
                        help='指定消融组，可重复指定；默认执行 core 单模块消融，all 选择历史四组，module 为兼容补充组')
    parser.add_argument('--skip-seg', action='store_true',
                        help='跳过房间分割')
    parser.add_argument('--image-dir', type=str, default='input/photo',
                        help='户型图目录')
    parser.add_argument('--traj-dir', type=str, default='input/traj',
                        help='轨迹数据目录')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='消融实验输出目录')

    args = parser.parse_args()

    if args.code:
        codes = args.code
    else:
        codes = [c.strip() for c in args.codes.split(',') if c.strip()]

    api_key = os.environ.get("FURNITURE_API_KEY")
    base_url = os.environ.get("FURNITURE_BASE_URL")
    if not api_key:
        print("[错误] 环境变量 FURNITURE_API_KEY 未设置")
        sys.exit(1)

    gt_dir = os.path.join(PROJECT_ROOT, 'GT')
    all_summary_lines = []

    for code in codes:
        gt_path = os.path.join(gt_dir, f'layout_{code}.yaml')
        if not os.path.exists(gt_path):
            print(f"[SKIP] GT 文件不存在: {gt_path}")
            continue

        image_path = os.path.join(PROJECT_ROOT, args.image_dir, f'room_{code}.png')
        if not os.path.exists(image_path):
            alt = os.path.join(PROJECT_ROOT, args.image_dir, f'room_{code[-2:]}.png')
            if os.path.exists(alt):
                image_path = alt
            else:
                print(f"[SKIP] 图片不存在: {image_path}")
                continue

        smart_device_path = os.path.join(
            PROJECT_ROOT, 'input', 'Smart_device', f'smartDevice_{code}.yaml')
        if not os.path.exists(smart_device_path):
            print(f"[WARN] 智能设备文件不存在，当前家庭不使用设备先验: {smart_device_path}")
            smart_device_path = None

        traj_dir = args.traj_dir
        if not os.path.isabs(traj_dir):
            traj_dir = os.path.join(PROJECT_ROOT, traj_dir)
        traj_dir = os.path.join(traj_dir, code)
        trajectory_path = None
        if os.path.isdir(traj_dir):
            jsons = sorted(Path(traj_dir).glob('*.json'),
                           key=lambda p: p.stat().st_size, reverse=True)
            if jsons:
                trajectory_path = str(jsons[0])

        output_dir = os.path.join(PROJECT_ROOT, args.output or f'output/{code}/ablation')
        code_summary = []

        selected_group_infos = _resolve_group_infos(args.group)

        for group_key, group_info in selected_group_infos:
            print(f"\n{'*'*60}")
            print(f"*  家庭 {code} — {group_info['name']}")
            print(f"{'*'*60}")

            results = run_ablation_group(
                code, group_key, group_info,
                image_path, smart_device_path, trajectory_path,
                output_dir, gt_path, args.skip_seg,
                api_key, base_url,
            )

            table_lines = _build_module_table(
                group_info["name"], group_info["variants"], results,
                group_info.get("description"))
            for line in table_lines:
                print(line)
            code_summary.extend(table_lines)

        all_summary_lines.extend(code_summary)

        # 保存本家庭消融结果
        eval_dir = os.path.join(PROJECT_ROOT, 'output', code, 'evaluation')
        os.makedirs(eval_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(eval_dir, f'ablation_results_{timestamp}.txt')
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(code_summary))
            f.write("  Semantic FDR = 1 - Semantic Precision\n")
        print(f"\n  消融实验结果已保存至: {save_path}")


if __name__ == '__main__':
    main()
