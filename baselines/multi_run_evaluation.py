"""
多轮评估脚本 — multi_run_evaluation.py

对每个户型运行 N 次完整 Pipeline，评估 LLM 随机性带来的输出波动。
报告 mean ± std，证明 Constrained Action Protocol 能有效控制方差。

用法:
  # 单个户型，5 轮
  python baselines/multi_run_evaluation.py --code 0622 --runs 5

  # 多个户型，10 轮
  python baselines/multi_run_evaluation.py --code 0622 --code 0701 --runs 10

  # 自定义输出目录
  python baselines/multi_run_evaluation.py --code 0622 --runs 5 --output output/0622/evaluation
"""

import os, sys, argparse, json, time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv
_env_path = os.path.join(PROJECT_ROOT, '.env')
if os.path.exists(_env_path):
    for enc in ('utf-8-sig', 'utf-16-le', 'utf-16', 'utf-8'):
        try:
            load_dotenv(dotenv_path=_env_path, encoding=enc)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

from langchain_core.callbacks import BaseCallbackHandler

from pipeline import Pipeline
from llm_config import DEFAULT_LLM_MODEL
from evaluation import (evaluate_single, load_yaml, parse_rooms, match_rooms,
                        Furniture, hungarian_matching)


PIPELINE_MODEL = DEFAULT_LLM_MODEL
PRICING_REFERENCE = (
    "repository cost proxy ($2.50 input / $10.00 output per 1M tokens); "
    "verify endpoint pricing before publication"
)


# ============================================================
#  API 调用追踪器
# ============================================================

class APITracker(BaseCallbackHandler):
    """追踪 LLM API 调用次数和 token 用量。"""

    def __init__(self):
        self.call_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def on_llm_end(self, response, **kwargs):
        self.call_count += 1
        llm_output = response.llm_output or {}
        usage = llm_output.get('token_usage', {}) or {}
        self.prompt_tokens += usage.get('prompt_tokens', 0)
        self.completion_tokens += usage.get('completion_tokens', 0)
        self.total_tokens += usage.get('total_tokens', 0)

    @property
    def stats(self) -> dict:
        return {
            "api_calls": self.call_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def _estimate_cost(prompt_tokens: int, completion_tokens: int,
                   model: str = PIPELINE_MODEL) -> float:
    """使用仓库代理费率估算成本；正式报告前需核对服务端价格。"""
    _ = model  # Kept for call-site compatibility and result metadata.
    p_in, p_out = 2.50, 10.00
    return (prompt_tokens / 1_000_000) * p_in + (completion_tokens / 1_000_000) * p_out


# ============================================================
#  核心: 单轮运行 + 评估
# ============================================================

def run_single_and_evaluate(
    code: str,
    image_path: str,
    smart_device_path: str,
    trajectory_path: str,
    hatch_mask_path: str,
    gt_path: str,
    output_dir: str,
    run_id: int,
    skip_seg: bool,
    skip_agents: bool,
    api_key: str,
    base_url: str,
) -> dict:
    """运行一次完整 Pipeline 并评估，返回指标字典（含耗时和 API 统计）。"""
    run_label = f"run_{run_id:02d}"
    run_out = os.path.join(output_dir, run_label)
    os.makedirs(run_out, exist_ok=True)

    tracker = APITracker()

    pipeline = Pipeline(
        api_key=api_key, base_url=base_url,
        ablation={},
        code=code,
        callbacks=[tracker] if not skip_agents else None,
    )

    t0 = time.perf_counter()
    pred_path = pipeline.run(
        image_path=image_path,
        smart_device_path=smart_device_path,
        trajectory_path=trajectory_path,
        output_dir=run_out,
        skip_seg=skip_seg,
        skip_agents=skip_agents,
    )
    elapsed = time.perf_counter() - t0

    result_a, result_b = evaluate_single(pred_path, gt_path)
    r = result_a

    api = tracker.stats

    return {
        "run": run_id,
        "rta": r.room_type_accuracy,
        "room_f1": r.room_f1,
        "f1": r.furniture_f1,
        "semantic_f1": r.semantic_f1,
        "fna": r.furniture_naming_accuracy,
        "cls": r.cls,
        "precision": r.furniture_precision,
        "recall": r.furniture_recall,
        "centroid_f1": r.centroid_f1,
        "centroid_recall": r.centroid_recall,
        "pred_path": pred_path,
        "elapsed_sec": round(elapsed, 1),
        "api_calls": api["api_calls"],
        "prompt_tokens": api["prompt_tokens"],
        "completion_tokens": api["completion_tokens"],
        "total_tokens": api["total_tokens"],
        "model": PIPELINE_MODEL,
        "endpoint": base_url or "OpenAI default endpoint",
        "pricing_reference": PRICING_REFERENCE,
    }


# ============================================================
#  表格输出
# ============================================================

def _mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.array(values)
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def print_results_table(
    per_code_results: Dict[str, List[dict]],
    num_runs: int,
) -> List[str]:
    """打印并返回结果文本行列表（含指标 + 性能开销）。"""
    lines = []
    lines.append(f"{'='*100}")
    lines.append(f"  多轮评估结果 — LLM 输出稳定性 + 性能开销")
    lines.append(f"  每户型运行次数: {num_runs}")
    lines.append(f"{'='*100}")

    # ── 指标表 ──
    header = (f"  {'家庭':<8} {'RoomF1':>16} {'LocF1':>16} {'SemF1':>16} {'CondFNA':>16} "
              f"{'LegacyCLS':>16} {'CLS Range':>12}")
    sep = f"  {'-'*95}"
    lines.append(header)
    lines.append(sep)

    for code, runs in sorted(per_code_results.items()):
        metrics = {}
        for key in ['room_f1', 'f1', 'semantic_f1', 'fna', 'cls']:
            vals = [r[key] for r in runs if r[key] is not None]
            if vals:
                metrics[key] = _mean_std(vals)
            else:
                metrics[key] = (0.0, 0.0)

        cls_vals = [r['cls'] for r in runs if r['cls'] is not None]
        cls_range = f"{min(cls_vals):.1%}~{max(cls_vals):.1%}" if cls_vals else "N/A"

        room_f1_s = f"{metrics['room_f1'][0]:.1%} ±{metrics['room_f1'][1]:.1%}"
        f1_s = f"{metrics['f1'][0]:.1%} ±{metrics['f1'][1]:.1%}"
        semantic_f1_s = f"{metrics['semantic_f1'][0]:.1%} ±{metrics['semantic_f1'][1]:.1%}"
        fna_s = f"{metrics['fna'][0]:.1%} ±{metrics['fna'][1]:.1%}"
        cls_s = f"{metrics['cls'][0]:.1%} ±{metrics['cls'][1]:.1%}"

        lines.append(
            f"  {code:<8} {room_f1_s:>16} {f1_s:>16} {semantic_f1_s:>16} "
            f"{fna_s:>16} {cls_s:>16} {cls_range:>12}"
        )

    lines.append(sep)

    # ── 性能开销表 ──
    lines.append(f"\n  {'─'*80}")
    lines.append(f"  性能开销 (per run avg)")
    lines.append(f"  {'─'*80}")
    perf_header = (f"  {'家庭':<8} {'耗时(秒)':>10} {'API调用':>8} "
                   f"{'Prompt Tok':>10} {'Compl Tok':>10} {'总Tokens':>10} {'成本(USD)':>10}")
    lines.append(perf_header)
    lines.append(f"  {'-'*70}")

    for code, runs in sorted(per_code_results.items()):
        elapsed = _mean_std([r.get('elapsed_sec', 0) for r in runs])
        api_calls = _mean_std([r.get('api_calls', 0) for r in runs])
        prompt_t = int(np.mean([r.get('prompt_tokens', 0) for r in runs]))
        compl_t = int(np.mean([r.get('completion_tokens', 0) for r in runs]))
        total_t = prompt_t + compl_t
        cost = _estimate_cost(prompt_t, compl_t)

        lines.append(
            f"  {code:<8} {elapsed[0]:>8.1f}±{elapsed[1]:.0f} "
            f"{api_calls[0]:>6.0f}±{api_calls[1]:.0f} "
            f"{prompt_t:>10,} {compl_t:>10,} {total_t:>10,} "
            f"${cost:>9.4f}"
        )

    lines.append(f"  {'-'*70}")
    observed_models = sorted({r.get('model', PIPELINE_MODEL)
                              for runs in per_code_results.values() for r in runs})
    observed_endpoints = sorted({r.get('endpoint', 'unknown')
                                 for runs in per_code_results.values() for r in runs})
    lines.append(f"  模型: {', '.join(observed_models)}; Endpoint: {', '.join(observed_endpoints)}")
    lines.append("  成本为估算值，采用仓库中的 2025 GPT-4o 假设价 ($2.50/$10.00 per 1M tokens)，发表前需核价。")

    # ── 跨户型汇总 ──
    if len(per_code_results) > 1:
        lines.append(f"\n  {'─'*60}")
        lines.append(f"  跨户型汇总 (N={len(per_code_results)} 个家庭)")
        lines.append(f"  {'─'*60}")
        for key, label in [('room_f1', 'Room F1'), ('f1', 'Localization F1'),
                           ('semantic_f1', 'Semantic F1'), ('fna', 'Conditional FNA')]:
            means = []
            stds = []
            for code, runs in per_code_results.items():
                vals = [r[key] for r in runs if r[key] is not None]
                if vals:
                    m, s = _mean_std(vals)
                    means.append(m)
                    stds.append(s)
            if means:
                avg_mean = float(np.mean(means))
                avg_std = float(np.mean(stds))
                lines.append(f"  {label}: {avg_mean:.1%} ±{avg_std:.1%} (avg across homes)")

        # 总计开销
        total_elapsed = sum(
            sum(r.get('elapsed_sec', 0) for r in runs)
            for runs in per_code_results.values()
        )
        total_calls = sum(
            sum(r.get('api_calls', 0) for r in runs)
            for runs in per_code_results.values()
        )
        total_prompt = sum(
            sum(r.get('prompt_tokens', 0) for r in runs)
            for runs in per_code_results.values()
        )
        total_compl = sum(
            sum(r.get('completion_tokens', 0) for r in runs)
            for runs in per_code_results.values()
        )
        total_cost = sum(
            _estimate_cost(
                sum(r.get('prompt_tokens', 0) for r in runs),
                sum(r.get('completion_tokens', 0) for r in runs),
            )
            for runs in per_code_results.values()
        )
        lines.append(f"\n  {'─'*60}")
        lines.append(f"  总计开销 (所有户型 × 所有轮次)")
        lines.append(f"  {'─'*60}")
        lines.append(f"  总耗时:     {total_elapsed:.0f} 秒 ({total_elapsed/60:.1f} 分钟)")
        lines.append(f"  API 调用:   {total_calls}")
        lines.append(f"  Prompt:     {total_prompt:,} tokens")
        lines.append(f"  Completion: {total_compl:,} tokens")
        lines.append(f"  总成本:     ${total_cost:.4f}")

    # ── 稳定性分析 ──
    cls_std_all = []
    for code, runs in per_code_results.items():
        vals = [r['cls'] for r in runs if r['cls'] is not None]
        if len(vals) >= 2:
            _, s = _mean_std(vals)
            cls_std_all.append((code, s))

    lines.append(f"\n  {'─'*60}")
    lines.append(f"  聚合分数波动 (per-home legacy CLS std)")
    lines.append(f"  {'─'*60}")
    for code, std in sorted(cls_std_all, key=lambda x: x[1]):
        lines.append(f"  {code}: CLS σ = {std:.3f} ({std:.1%})")

    avg_cls_std = float(np.mean([s for _, s in cls_std_all])) if cls_std_all else 0
    lines.append(f"  平均 CLS σ = {avg_cls_std:.3f} ({avg_cls_std:.1%})")
    lines.append("  解读限制: 聚合分数方差小不等于实例预测一致；应与下方逐家具一致率联合报告。")

    # ── 逐家具命名一致性 ──
    lines.extend(_furniture_naming_consistency(per_code_results))

    return lines


def _furniture_naming_consistency(per_code_results: Dict[str, List[dict]]) -> List[str]:
    """分析每个家具在不同轮次间的命名一致性，含房间位置和 GT 标签。

    GT 辅助对齐只用于给预测实例附加可读标签，不参与主性能计算。
    """
    lines = []
    for code, runs in per_code_results.items():
        lines.append(f"\n  {'─'*110}")
        lines.append(f"  逐家具命名一致性分析 ({code})")
        lines.append(f"  {'─'*110}")

        # ---- 加载并解析 GT 房间 ----
        gt_rooms = []
        gt_path = os.path.join(PROJECT_ROOT, 'GT', f'layout_{code}.yaml')
        if os.path.exists(gt_path):
            gt_rooms = parse_rooms(load_yaml(gt_path))

        # ---- 收集每轮每件家具的名字 ----
        furniture_info = {}  # key: "room_label/fur_id" → {names, pos, gt}

        for run_data in runs:
            pred_path = run_data.get('pred_path', '')
            if not pred_path or not os.path.exists(pred_path):
                continue

            # 加载预测原始数据 (保留 furniture id)
            pred_raw = load_yaml(pred_path)

            # 用 parse_rooms 解析房间/家具 (不含 id)
            pred_rooms = parse_rooms(pred_raw)

            # 手动提取每个预测家具的 ID (按与 parse_rooms 相同的遍历顺序)
            pred_ids_by_room = []  # List[List[str]]: 每个房间的家具 ID 列表
            house_data = pred_raw.get('house', pred_raw)
            for room_data in house_data.get('rooms', []):
                fur_ids = []
                for fur_data in room_data.get('furniture', []):
                    fur_ids.append(fur_data.get('id', ''))
                pred_ids_by_room.append(fur_ids)

            # ---- 策略 B: 房间对齐 + 家具匈牙利匹配 ----
            gt_name_map = {}  # (pred_room_idx, pred_fur_idx) → gt_furniture_name
            if gt_rooms:
                room_match_map, _ = match_rooms(gt_rooms, pred_rooms)
                for gi, pj in room_match_map.items():
                    gt_room = gt_rooms[gi]
                    pred_room = pred_rooms[pj]
                    gt_furs = gt_room.furniture
                    pred_furs = pred_room.furniture
                    if not gt_furs or not pred_furs:
                        continue

                    # 房间中心偏移
                    dx = gt_room.bbox.center_x - pred_room.bbox.center_x
                    dy = gt_room.bbox.center_y - pred_room.bbox.center_y

                    # 对齐预测家具
                    aligned_preds = [
                        Furniture(f.name, f.bbox.translate(dx, dy), f.room_name)
                        for f in pred_furs
                    ]

                    n_gt = len(gt_furs)
                    n_pred = len(aligned_preds)
                    cost = [[0.0] * n_pred for _ in range(n_gt)]
                    for i in range(n_gt):
                        for j in range(n_pred):
                            cost[i][j] = 1.0 - gt_furs[i].bbox.iou(aligned_preds[j].bbox)

                    raw_matches = hungarian_matching(cost)
                    for fi, fj in raw_matches:
                        gt_name_map[(pj, fj)] = gt_furs[fi].name

            # ---- 遍历预测数据，收集命名信息 ----
            for pj, room_data in enumerate(house_data.get('rooms', [])):
                room_name = room_data.get('name', '?')
                room_pos = room_data.get('position', {})
                room_label = (f"{room_name}"
                              f"({room_pos.get('x',0):.0f},{room_pos.get('y',0):.0f})")
                fur_ids = pred_ids_by_room[pj]
                for fj, fur_data in enumerate(room_data.get('furniture', [])):
                    fur_name = (fur_data.get('name', '') or '').strip()
                    fur_id = fur_data.get('id', '')
                    fur_pos = fur_data.get('position', {})
                    if not fur_name or not fur_id:
                        continue
                    pos_str = (f"({fur_pos.get('x',0):.0f},{fur_pos.get('y',0):.0f})"
                               f" {fur_pos.get('width',0):.0f}x{fur_pos.get('height',0):.0f}")
                    key = f"{room_label}/{fur_id}"
                    if key not in furniture_info:
                        gt_match = gt_name_map.get((pj, fj), "?")
                        furniture_info[key] = {
                            'names': Counter(),
                            'pos': pos_str,
                            'gt': gt_match,
                        }
                    furniture_info[key]['names'][fur_name] += 1

        # 找出不一致的家具
        inconsistent = []
        for key, info in sorted(furniture_info.items()):
            if len(info['names']) > 1:
                inconsistent.append((key, info['pos'], info['gt'], info['names']))

        n_instances = len(furniture_info)
        exact_consistency = ((n_instances - len(inconsistent)) / n_instances
                             if n_instances else 0.0)
        total_observations = sum(sum(info['names'].values()) for info in furniture_info.values())
        majority_agreement = (sum(max(info['names'].values()) for info in furniture_info.values()) /
                              total_observations if total_observations else 0.0)
        lines.append(f"  完全一致实例: {n_instances - len(inconsistent)}/{n_instances} "
                     f"({exact_consistency:.1%})")
        lines.append(f"  多数票一致率: {majority_agreement:.1%} (仅对各轮已出现的实例计数)")

        if inconsistent:
            lines.append(f"  {'房间/家具ID':<42} {'坐标(w×h)':<17} {'GT名称':<16} {'各轮名字':<40}")
            lines.append(f"  {'-'*115}")
            for key, pos, gt, names in sorted(inconsistent, key=lambda x: -len(x[3])):
                names_text = ', '.join(f"{name}×{count}" for name, count in names.most_common())
                lines.append(f"  {key:<42} {pos:<17} {gt:<16} {names_text:<40}")
            lines.append(f"\n  共 {len(inconsistent)} 件家具在不同轮次间名字不一致")
        else:
            lines.append("  所有已跟踪家具的命名完全一致")

    return lines


# ============================================================
#  主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='多轮评估：运行 N 次 Pipeline，评估 LLM 输出稳定性')
    parser.add_argument('--code', type=str, action='append', default=None,
                        help='家庭编码 (可多次指定，默认: 0622)')
    parser.add_argument('--runs', type=int, default=5,
                        help='每户型运行次数 (默认: 5)')
    parser.add_argument('--skip-seg', action='store_true', default=True,
                        help='跳过房间分割 (默认开启，分割是确定性的)')
    parser.add_argument('--no-skip-seg', dest='skip_seg', action='store_false',
                        help='强制运行房间分割')
    parser.add_argument('--skip-agents', action='store_true',
                        help='跳过所有 LLM Agents (仅测试 CV 管线)')
    parser.add_argument('--image-dir', type=str, default='input/photo',
                        help='户型图目录')
    parser.add_argument('--traj-dir', type=str, default='input/traj',
                        help='轨迹数据目录')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='输出目录 (默认: output/{code}/multi_run_eval)')

    args = parser.parse_args()

    if args.code is None:
        codes = ['0622']
    else:
        codes = args.code

    num_runs = args.runs

    api_key = os.environ.get("FURNITURE_API_KEY")
    base_url = os.environ.get("FURNITURE_BASE_URL")
    if not api_key:
        print("[错误] 环境变量 FURNITURE_API_KEY 未设置")
        sys.exit(1)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    all_results = {}
    for code in codes:
        print(f"\n{'#'*60}")
        print(f"#  多轮评估: {code} — {num_runs} runs")
        print(f"{'#'*60}")

        # 验证输入文件
        gt_path = os.path.join(PROJECT_ROOT, 'GT', f'layout_{code}.yaml')
        image_path = os.path.join(PROJECT_ROOT, args.image_dir, f'room_{code}.png')
        smart_device_path = os.path.join(PROJECT_ROOT, 'input', 'Smart_device',
                                         f'smartDevice_{code}.yaml')
        traj_dir = os.path.join(PROJECT_ROOT, args.traj_dir, code)

        for label, path in [('GT', gt_path), ('户型图', image_path),
                            ('智能设备', smart_device_path)]:
            if not os.path.exists(path):
                print(f"[错误] {label} 文件不存在: {path}")
                sys.exit(1)

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
                print(f"[警告] hatch_mask.png 不存在")
                hatch_mask_path = None

        base_out = args.output or os.path.join(PROJECT_ROOT, 'output', code, 'multi_run_eval')
        output_dir = os.path.join(base_out, f'multi_run_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)

        code_runs = []
        for run_id in range(1, num_runs + 1):
            print(f"\n  ── Run {run_id}/{num_runs} ──")
            try:
                metrics = run_single_and_evaluate(
                    code=code,
                    image_path=image_path,
                    smart_device_path=smart_device_path,
                    trajectory_path=trajectory_path,
                    hatch_mask_path=hatch_mask_path,
                    gt_path=gt_path,
                    output_dir=output_dir,
                    run_id=run_id,
                    skip_seg=args.skip_seg,
                    skip_agents=args.skip_agents,
                    api_key=api_key,
                    base_url=base_url,
                )
                code_runs.append(metrics)
                print(f"    CLS={metrics['cls']:.1%}  RTA={metrics['rta']:.1%}  "
                      f"F1={metrics['f1']:.1%}  FNA={metrics['fna']:.1%}  "
                      f"{metrics['elapsed_sec']:.0f}s  {metrics['api_calls']} calls")
                # 轮间冷却，避免 API 限流
                if run_id < num_runs:
                    time.sleep(5)
            except Exception as e:
                print(f"    [ERROR] Run {run_id}: {e}")
                import traceback
                traceback.print_exc()

        if not code_runs:
            print(f"  [ERROR] {code}: 所有 {num_runs} 轮均失败，跳过")
            continue

        all_results[code] = code_runs

    # ── 输出结果 ──
    print()
    lines = print_results_table(all_results, num_runs)

    # 保存
    eval_dir = os.path.join(PROJECT_ROOT, 'output', codes[0], 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    txt_path = os.path.join(eval_dir, f'multi_run_eval_{timestamp}.txt')
    json_path = os.path.join(eval_dir, f'multi_run_eval_{timestamp}.json')

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    # 结构化 JSON
    structured = {}
    for code, runs in all_results.items():
        structured[code] = {
            "num_runs": num_runs,
            "runs": runs,
            "aggregated": {
                key: list(_mean_std([r[key] for r in runs if r[key] is not None]))
                for key in ['rta', 'f1', 'fna', 'cls', 'centroid_f1', 'centroid_recall']
            } if runs else {}
        }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(structured, f, indent=2, ensure_ascii=False)

    print(f"\n文本结果已保存: {txt_path}")
    print(f"结构化数据已保存: {json_path}")
    print(f"\n{'='*60}")
    print(f"  多轮评估完成")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
