"""
消融实验主脚本 — ablation_experiments.py

UbiComp 精简版：2 组核心消融，每组 ≤ 1 张表，总共 ≤ 1 页。

  1. 模态贡献消融 (Modal Contribution) — 必须
     累加式验证各模态的独立与互补贡献
     (Layout Only → +Device → +Trajectory → Full)
     重点: Device 提升语义 (RTA/FNA), Trajectory 提供行为信息

  2. Constrained vs Free-form LLM — 强烈建议
     验证约束协议减少词表协议违规，并检查端到端语义误报
     Free-form: 真正无约束 (去掉 allowed_list + shape_constraint)
     Protocol Violation: per-room allowed list（不读取 GT）

用法:
  # 全部 2 组
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


# ============================================================
# 协议违规率计算
# ============================================================

FURNITURE_ALLOWED = load_furniture_allowed()
SMART_DEVICE_NAMES = get_smart_device_names()

def _normalize_fur_name(name: str) -> str:
    """家具名规范化 (与 evaluation 对齐)"""
    mapping = {
        'coffee table': 'coffee_table', 'coffee_table': 'coffee_table',
        'dining table': 'dining_table', 'dining_table': 'dining_table',
        'shoe cabinet': 'shoe_cabinet', 'shoe_cabinet': 'shoe_cabinet',
        'coat rack': 'coat_rack', 'coat_rack': 'coat_rack',
        'washing machine': 'washing_machine',
        'tv cabinet': 'tv_cabinet',
        'tv stand': 'tv_stand',
        'night stand': 'nightstand',
        'book shelf': 'bookshelf',
    }
    key = name.lower().replace('_', ' ').strip()
    return mapping.get(key, key.replace(' ', '_'))


def compute_protocol_violation_rate(pred_yaml_path: str) -> tuple:
    """计算非智能设备预测违反房间允许词表的比例。

    Returns:
        (violation_rate, n_violations, n_non_device_predictions)

    这是约束机制检查，不是端到端性能指标。语义误报应使用
    ``1 - EvaluationResult.semantic_precision``。
    """
    if not os.path.exists(pred_yaml_path):
        return 0.0, 0, 0

    with open(pred_yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    rooms = data.get('house', {}).get('rooms', [])
    n_total = 0
    n_violations = 0
    smart_names = {_normalize_fur_name(name) for name in SMART_DEVICE_NAMES}

    for room in rooms:
        room_type = (room.get('name', '') or '').lower()
        allowed = FURNITURE_ALLOWED.get(room_type, FURNITURE_ALLOWED.get('other', []))
        allowed_normalized = {_normalize_fur_name(name) for name in allowed}

        for fur in room.get('furniture', []):
            fur_name = (fur.get('name', '') or '').strip()
            if not fur_name:
                continue
            normalized = _normalize_fur_name(fur_name)
            # 智能设备是输入先验，不进入该机制指标的分子或分母。
            if normalized in smart_names:
                continue
            n_total += 1
            if normalized not in allowed_normalized:
                n_violations += 1

    if n_total == 0:
        return 0.0, 0, 0
    return n_violations / n_total, n_violations, n_total


def compute_hallucination_rate(pred_yaml_path: str, gt_path: str = None) -> tuple:
    """Backward-compatible alias; GT is intentionally ignored."""
    return compute_protocol_violation_rate(pred_yaml_path)


# ============================================================
# 消融实验变体定义 (UbiComp 精简版)
# ============================================================

# ── 1. 模态贡献消融 (累加式) ──
# 验证各模态的独立贡献与互补性
# trajectory 贡献重点: 行为信息 (非整体 CLS)
MODAL_VARIANTS = {
    "Layout Only":              {"skip_trajectory": True, "skip_smart_device": True},
    "+ Device":                 {"skip_trajectory": True},
    "+ Trajectory":             {"skip_smart_device": True},
    "Full (All Modalities)":    {},
}

# ── 2. Constrained vs Free-form LLM ──
# Free-form: 真正无约束 LLM (去掉 allowed_list, shape_constraint)
# 验证约束协议减少词表协议违规，并检查端到端语义误报
LLM_VARIANTS = {
    "Free-form LLM":            {"use_free_form_llm": True,
                                 "skip_allowed_list": True,
                                 "skip_shape_constraint": True},
    "Constrained (Ours)":       {},
}

# ── 分组定义 ──
ABLATION_GROUPS = {
    "modal": {
        "name": "Modal Contribution",
        "variants": MODAL_VARIANTS,
        "table_type": "modal",  # 累加式表格，含 Protocol Violation Rate
    },
    "llm": {
        "name": "Constrained vs Free-form LLM",
        "variants": LLM_VARIANTS,
        "table_type": "llm",    # 协议违规率 + Semantic FDR
    },
}


# ============================================================
# 辅助函数
# ============================================================

def run_pipeline_with_ablation(
    code: str, ablation: dict, variant_name: str,
    image_path: str, smart_device_path: str, trajectory_path: str,
    output_dir: str, skip_seg: bool = False,
    api_key: str = None, base_url: str = None,
) -> str:
    """运行带消融配置的 Pipeline，返回最终 YAML 路径"""
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
    for variant_name, ablation_config in variants.items():
        try:
            # 所有变体（包括 Full）都在同一实验批次重新运行，避免复用由
            # 不同代码、模型或 prompt 版本产生的旧结果。
            pred_path = run_pipeline_with_ablation(
                code, ablation_config, variant_name,
                image_path, smart_device_path, trajectory_path,
                output_dir, skip_seg, api_key, base_url,
            )
            result_a, result_b = evaluate_single(pred_path, gt_path)
            protocol_rate, protocol_n, protocol_total = compute_protocol_violation_rate(pred_path)
            results[variant_name] = {
                "result_a": result_a,
                "result_b": result_b,
                # 保留旧 key，兼容现有绘图脚本。
                "hallu_rate": protocol_rate,
                "hallu_n": protocol_n,
                "hallu_total": protocol_total,
                "pred_path": pred_path,
            }
        except Exception as e:
            print(f"  [ERROR] {variant_name}: {e}")
            import traceback
            traceback.print_exc()
            results[variant_name] = None

    return results


def _build_modal_table(group_name: str, variants: dict, results: dict) -> list:
    """构建模态贡献表格"""
    lines = []
    lines.append(f"{'='*95}")
    lines.append(f"  {group_name}  (独立与叠加贡献)")
    lines.append(f"{'='*95}")

    header = (f"  {'Variant':<25} {'RTA':>7} {'F1':>7} "
              f"{'FNA':>7} {'CLS':>7} {'ProtV':>7}")
    sep = f"  {'-'*65}"
    lines.append(header)
    lines.append(sep)

    for variant_name in variants.keys():
        result = results.get(variant_name)
        if result is None:
            lines.append(f"  {variant_name:<25} {'ERROR':>7}")
            continue
        r = result.get("result_a")
        if r is None or r.furniture_f1 is None:
            lines.append(f"  {variant_name:<25} {'N/A':>7} {'N/A':>7} {'N/A':>7} {'N/A':>7} {'N/A':>7}")
            continue

        hallu = result.get("hallu_rate", 0.0)
        lines.append(f"  {variant_name:<25} {r.room_type_accuracy:>6.1%} {r.furniture_f1:>6.1%} "
                     f"{r.furniture_naming_accuracy:>6.1%} {r.cls:>6.1%} {hallu:>6.1%}")

    lines.append(sep)

    # ── 各模态独立贡献 (按评价维度拆解) ──
    layout = results.get("Layout Only")
    device = results.get("+ Device")
    traj = results.get("+ Trajectory")
    full = results.get("Full (All Modalities)")

    def _get_vals(r):
        if r is None: return None
        rr = r.get("result_a")
        if rr is None: return None
        return {
            "rta": rr.room_type_accuracy,
            "f1": rr.furniture_f1,
            "fna": rr.furniture_naming_accuracy,
            "cls": rr.cls,
        }

    lv, dv, tv, fv = _get_vals(layout), _get_vals(device), _get_vals(traj), _get_vals(full)

    lines.append("  --- 各模态独立贡献 (per dimension) ---")
    lines.append(f"  说明: Device 提升语义 (RTA/FNA), Trajectory 提供行为信息")

    # Device 单独贡献
    if lv and dv:
        lines.append(f"  +Device (vs Layout):  RTA {lv['rta']:.1%}→{dv['rta']:.1%}  "
                     f"FNA {lv['fna']:.1%}→{dv['fna']:.1%}  "
                     f"CLS {lv['cls']:.1%}→{dv['cls']:.1%}")
    # Trajectory 单独贡献
    if lv and tv:
        lines.append(f"  +Trajectory (vs Layout):  "
                     f"CLS {lv['cls']:.1%}→{tv['cls']:.1%}")

    lines.append("  --- 互补叠加贡献 ---")
    if dv and fv:
        lines.append(f"  Full (vs +Device):  "
                     f"CLS {dv['cls']:.1%}→{fv['cls']:.1%}")
    if tv and fv:
        lines.append(f"  Full (vs +Trajectory):  "
                     f"CLS {tv['cls']:.1%}→{fv['cls']:.1%}")

    return lines


def _build_llm_table(group_name: str, variants: dict, results: dict) -> list:
    """构建 LLM 消融表格（协议违规率 + 端到端语义误报率）。"""
    lines = []
    lines.append(f"{'='*95}")
    lines.append(f"  {group_name}")
    lines.append("  ProtV: 非智能设备名称不在房间允许词表；Semantic FDR = 1 - Semantic Precision")
    lines.append(f"{'='*95}")

    header = (f"  {'Variant':<25} {'ProtV':>8} {'#Viol':>8} {'SemFDR':>8} "
              f"{'RTA':>7} {'F1':>7} {'FNA':>7} {'CLS':>7}")
    sep = f"  {'-'*75}"
    lines.append(header)
    lines.append(sep)

    for variant_name in variants.keys():
        result = results.get(variant_name)
        if result is None:
            lines.append(f"  {variant_name:<25} {'ERROR':>11}")
            continue
        r = result.get("result_a")
        hallu_rate = result.get("hallu_rate", 0.0)
        hallu_n = result.get("hallu_n", 0)
        hallu_total = result.get("hallu_total", 0)
        hallu_str = f"{hallu_rate:.1%}"
        hallu_count = f"{hallu_n}/{hallu_total}"
        semantic_fdr = 1.0 - r.semantic_precision if r is not None else 0.0

        if r is None or r.furniture_f1 is None:
            lines.append(f"  {variant_name:<25} {hallu_str:>8} {hallu_count:>8} {'N/A':>8} "
                         f"{'N/A':>7} {'N/A':>7} {'N/A':>7} {'N/A':>7}")
            continue

        lines.append(f"  {variant_name:<25} {hallu_str:>8} {hallu_count:>8} {semantic_fdr:>7.1%} "
                     f"{r.room_type_accuracy:>6.1%} {r.furniture_f1:>6.1%} "
                     f"{r.furniture_naming_accuracy:>6.1%} {r.cls:>6.1%}")

    lines.append(sep)

    # 协议违规率降幅 + FNA 变化
    free_result = None
    for vn in variants.keys():
        if "free" in vn.lower():
            free_result = results.get(vn)
            break

    constrained_result = results.get("Constrained (Ours)", None)
    if free_result and constrained_result:
        free_h = free_result.get("hallu_rate", 0.0)
        cons_h = constrained_result.get("hallu_rate", 0.0)
        free_r = free_result.get("result_a")
        cons_r = constrained_result.get("result_a")
        if free_h > 0:
            reduction = (free_h - cons_h) / free_h
            lines.append(f"  Protocol-violation reduction: {reduction:.1%} "
                         f"({free_h:.1%} → {cons_h:.1%})")
        if free_r and cons_r and free_r.furniture_naming_accuracy is not None:
            lines.append(f"  FNA improvement: "
                         f"{free_r.furniture_naming_accuracy:.1%} → {cons_r.furniture_naming_accuracy:.1%}")

    return lines


TABLE_BUILDERS = {
    "modal": _build_modal_table,
    "llm": _build_llm_table,
}


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='消融实验 — UbiComp 精简版 (2 组核心消融)')
    parser.add_argument('--code', type=str, action='append', default=None,
                        help='家庭编码 (如 0622)，可多次指定')
    parser.add_argument('--codes', type=str, default='0622',
                        help='用逗号分隔的编码列表 (如 0622,0701)')
    parser.add_argument('--group', type=str, action='append', default=None,
                        choices=list(ABLATION_GROUPS.keys()) + ['all'],
                        help='执行指定消融组 (可多次指定): modal, llm, all')
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

    if args.group is None or 'all' in args.group:
        groups = list(ABLATION_GROUPS.keys())
    else:
        groups = args.group

    api_key = os.environ.get("FURNITURE_API_KEY")
    base_url = os.environ.get("FURNITURE_BASE_URL")
    if not api_key:
        print("[错误] 环境变量 FURNITURE_API_KEY 未设置")
        sys.exit(1)

    gt_dir = 'GT'
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

        smart_device_path = os.path.join('input', 'Smart_device', f'smartDevice_{code}.yaml')
        if not os.path.exists(smart_device_path):
            sd_alt = os.path.join('input', 'Smart_device', 'smartDevice_0622.yaml')
            if os.path.exists(sd_alt):
                smart_device_path = sd_alt
            else:
                print(f"[WARN] 智能设备文件不存在: {smart_device_path}")

        traj_dir = os.path.join(args.traj_dir, code)
        trajectory_path = None
        if os.path.isdir(traj_dir):
            jsons = sorted(Path(traj_dir).glob('*.json'),
                           key=lambda p: p.stat().st_size, reverse=True)
            if jsons:
                trajectory_path = str(jsons[0])

        output_dir = os.path.join(PROJECT_ROOT, args.output or f'output/{code}/ablation')
        code_summary = []

        for group_key in groups:
            group_info = ABLATION_GROUPS[group_key]
            print(f"\n{'*'*60}")
            print(f"*  家庭 {code} — {group_info['name']}")
            print(f"{'*'*60}")

            results = run_ablation_group(
                code, group_key, group_info,
                image_path, smart_device_path, trajectory_path,
                output_dir, gt_path, args.skip_seg,
                api_key, base_url,
            )

            builder = TABLE_BUILDERS.get(group_info.get("table_type", "modal"),
                                          _build_modal_table)
            table_lines = builder(group_info["name"], group_info["variants"],
                                  results)
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
            f.write(f"\n\n  CLS = 0.45 x FNA + 0.35 x F1_IoU + 0.20 x RTA\n")
            f.write("  ProtV = 词表违规的非智能设备预测 / 非智能设备预测总数\n")
            f.write("  Semantic FDR = 1 - Semantic Precision\n")
        print(f"\n  消融实验结果已保存至: {save_path}")


if __name__ == '__main__':
    main()
