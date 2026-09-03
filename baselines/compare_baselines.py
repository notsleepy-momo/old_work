"""
对比实验主脚本 — compare_baselines.py

参考对比实验设计:
  - VLM Zero-shot: 户型图直接给 VLM，一次 Prompt 输出完整 YAML
  - VLM Chain-of-Thought: 户型图 + 分步 Prompt (房间→家具→命名)
  - CV-Only: 仅使用 CV 模型，不使用 LLM
  - Ours: 完整 CV+LLM 混合管线

自动执行:
  1. VLM Zero-shot baseline (可选 --skip-vlm 跳过)
  2. VLM Chain-of-Thought baseline (可选 --skip-vlm 跳过)
  3. CV-Only baseline
  4. 对四种方法统一执行 Global 与 Aligned 评分
  5. 输出两套同协议对比结果表 (可复制到论文)

用法:
  # 完整对比 (包括调用 VLM，耗时较长)
  python baselines/compare_baselines.py --code 0622
  python baselines/compare_baselines.py --code 0622 --excel

  # 仅评分 (VLM 结果已生成，跳过 API 调用)
  python baselines/compare_baselines.py --code 0622 --skip-vlm

  # 批量对比多个家庭
  python baselines/compare_baselines.py --code 0622 --code 0701
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from evaluation import evaluate_single


METHODS_ORDER = [
    ('vlm_zeroshot', 'VLM Zero-shot'),
    ('vlm_cot', 'VLM Chain-of-Thought'),
    ('cv_only', 'CV-Only (No LLM)'),
    ('ours', 'Ours'),
]

PROTOCOLS_ORDER = [
    ('global', 'A. 全局匹配 (Global)'),
    ('aligned', 'B. 房间内对齐匹配 (Aligned)'),
]


def run_evaluation(pred_path: str, gt_path: str, method_name: str,
                   quiet: bool = False) -> tuple:
    """运行评估并返回 (result_a, result_b)"""
    if not os.path.exists(pred_path):
        print(f"  [SKIP] {method_name}: 预测文件不存在 {pred_path}")
        return None, None

    if not quiet:
        print(f"\n{'─'*55}")
        print(f"  评估: {method_name}")
        print(f"  预测: {pred_path}")
        print(f"  GT:   {gt_path}")

    try:
        result_a, result_b = evaluate_single(pred_path, gt_path)
    except Exception as exc:
        print(f"  [FAIL] {method_name}: 评分失败: {exc}")
        return None, None
    return result_a, result_b


def generate_vlm_prediction(command: list, candidate_path: str,
                            final_path: str) -> bool:
    """Promote only a prediction produced successfully by this subprocess."""
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        print(f"  [FAIL] VLM 子进程退出码: {completed.returncode}")
        return False
    if not os.path.isfile(candidate_path) or os.path.getsize(candidate_path) == 0:
        print("  [FAIL] VLM 子进程未生成有效候选文件")
        return False
    try:
        os.makedirs(os.path.dirname(os.path.abspath(final_path)), exist_ok=True)
        os.replace(candidate_path, final_path)
    except OSError as exc:
        print(f"  [FAIL] 无法发布本次 VLM 预测: {exc}")
        return False
    print(f"  [OK] 本次预测已冻结: {final_path}")
    return True


def evaluate_vlm_prediction(generation_succeeded: bool, pred_path: str,
                            gt_path: str, method_name: str) -> tuple:
    if not generation_succeeded:
        print(f"  [SKIP] {method_name} 本次生成失败；不评分磁盘旧结果")
        return None, None
    return run_evaluation(pred_path, gt_path, method_name, quiet=True)


def _build_comparison_lines(results: dict) -> list:
    """构建两种统一评分协议的对比结果文本行列表。"""
    lines = []
    lines.append(f"{'='*80}")
    lines.append(f"  对比实验结果 (Comparison with Baselines)")
    lines.append(f"{'='*80}")

    lines.append("  Prediction generation: no method reads test GT; predictions are frozen before scoring.")
    lines.append("  Scoring: A uses absolute coordinates; B matches rooms, then translates each room pair")
    lines.append("  before furniture matching. The evaluator applies both protocols to every method.")

    header = (f"  {'Method':<28} {'RoomF1':>8} {'LocF1':>8} "
              f"{'SemF1':>8} {'CondFNA':>9} {'SemTP':>6}")
    sep = f"  {'-'*77}"
    for protocol_key, protocol_label in PROTOCOLS_ORDER:
        lines.append(f"\n  Protocol: {protocol_label}")
        lines.append(header)
        lines.append(sep)
        for method_key, method_label in METHODS_ORDER:
            result = results.get(f'{method_key}_{protocol_key}')
            if result is None or result.furniture_f1 is None:
                lines.append(
                    f"  {method_label:<28} {'N/A':>8} {'N/A':>8} "
                    f"{'N/A':>8} {'N/A':>9} {'N/A':>6}")
                continue

            lines.append(
                f"  {method_label:<28} {result.room_f1:>7.1%} "
                f"{result.furniture_f1:>7.1%} {result.semantic_f1:>7.1%} "
                f"{result.furniture_naming_accuracy:>8.1%} "
                f"{result.tp_correct_name:>6}")
        lines.append(sep)

    lines.append("\n  Compare methods only within the same protocol; do not compare A rows with B rows.")

    return lines


def print_comparison_table(results: dict):
    """打印并保存对比结果表"""
    lines = _build_comparison_lines(results)
    for line in lines:
        print(line)


def save_comparison_table(results: dict, output_path: str):
    """保存对比结果表到文件"""
    lines = _build_comparison_lines(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n  对比结果已保存至: {output_path}")


def _results_to_rows(results: dict) -> list:
    """将 Global 和 Aligned 结果转换为 Excel 行数据。"""
    rows = []
    for protocol_key, protocol_label in PROTOCOLS_ORDER:
        for method_key, method_label in METHODS_ORDER:
            r = results.get(f'{method_key}_{protocol_key}')
            if r is None or r.furniture_f1 is None:
                rows.append((protocol_label, method_label,
                             None, None, None, None, None))
                continue
            rows.append((protocol_label, method_label,
                         r.room_f1, r.furniture_f1, r.semantic_f1,
                         r.furniture_naming_accuracy, r.tp_correct_name))
    return rows


def save_comparison_excel(all_results: dict, codes: list, output_path: str):
    """保存对比结果到 Excel (每个家庭一个 Sheet，多家庭额外一个 Summary Sheet)"""
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, Border, Side, PatternFill

    wb = Workbook()
    wb.remove(wb.active)

    header_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
    header_font_white = Font(bold=True, size=11, color='FFFFFF')
    thin_border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin'))
    pct_fmt = '0.0%'

    headers = ['Protocol', 'Method', 'Room F1', 'Localization F1',
               'Semantic F1', 'Conditional FNA', 'Semantic TP']

    for code in codes:
        if code not in all_results:
            continue
        ws = wb.create_sheet(title=f'Home {code}')
        results = all_results[code]

        # 写入表头
        for col, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=h)
            cell.font = header_font_white
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal='center')
            cell.border = thin_border

        rows = _results_to_rows(results)
        for r_idx, row in enumerate(rows, 2):
            protocol, method, room_f1, loc_f1, sem_f1, fna, sem_tp = row
            ws.cell(row=r_idx, column=1, value=protocol).border = thin_border
            ws.cell(row=r_idx, column=2, value=method).border = thin_border
            ws.cell(row=r_idx, column=2).font = Font(bold=True)
            for c_idx, val in enumerate([room_f1, loc_f1, sem_f1, fna, sem_tp], 3):
                cell = ws.cell(row=r_idx, column=c_idx)
                if val is not None:
                    cell.value = val
                    cell.number_format = '0' if c_idx == 7 else pct_fmt
                else:
                    cell.value = 'N/A'
                cell.alignment = Alignment(horizontal='center')
                cell.border = thin_border

        # 列宽
        ws.column_dimensions['A'].width = 32
        ws.column_dimensions['B'].width = 28
        for c in 'CDEFG':
            ws.column_dimensions[c].width = 14

        last_row = len(rows) + 2
        ws.cell(row=last_row, column=1,
                value='Predictions are frozen before GT enters the independent scorer. Compare methods only within the same protocol.').font = Font(italic=True, size=9)

    # 多家庭时添加 Summary Sheet
    if len(codes) > 1:
        ws = wb.create_sheet(title='Summary')
        for col, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=h)
            cell.font = header_font_white
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal='center')
            cell.border = thin_border

        summary_specs = [
            (protocol_key, protocol_label, method_key, method_label)
            for protocol_key, protocol_label in PROTOCOLS_ORDER
            for method_key, method_label in METHODS_ORDER
        ]

        for r_idx, (protocol_key, protocol_label, method_key, method_label) in enumerate(summary_specs, 2):
            vals = []
            for code in codes:
                r = all_results.get(code, {}).get(f'{method_key}_{protocol_key}')
                if r and r.cls is not None:
                    vals.append(r)
            ws.cell(row=r_idx, column=1, value=protocol_label).border = thin_border
            ws.cell(row=r_idx, column=2, value=method_label).border = thin_border
            ws.cell(row=r_idx, column=2).font = Font(bold=True)
            if vals:
                avg_room_f1 = sum(v.room_f1 for v in vals) / len(vals)
                avg_f1 = sum(v.furniture_f1 for v in vals) / len(vals)
                avg_sem_f1 = sum(v.semantic_f1 for v in vals) / len(vals)
                avg_fna = sum(v.furniture_naming_accuracy for v in vals) / len(vals)
                avg_sem_tp = sum(v.tp_correct_name for v in vals) / len(vals)
                for c_idx, val in enumerate([avg_room_f1, avg_f1, avg_sem_f1, avg_fna, avg_sem_tp], 3):
                    cell = ws.cell(row=r_idx, column=c_idx, value=val)
                    cell.number_format = '0.0' if c_idx == 7 else pct_fmt
                    cell.alignment = Alignment(horizontal='center')
                    cell.border = thin_border
            else:
                for c_idx in range(3, 8):
                    cell = ws.cell(row=r_idx, column=c_idx, value='N/A')
                    cell.alignment = Alignment(horizontal='center')
                    cell.border = thin_border

        ws.column_dimensions['A'].width = 32
        ws.column_dimensions['B'].width = 28
        for c in 'CDEFG':
            ws.column_dimensions[c].width = 14

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    wb.save(output_path)
    print(f"  对比结果 Excel 已保存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='对比实验 — Ours vs VLM Baselines (参考论文实验设计)')
    parser.add_argument('--code', type=str, action='append', default=None,
                        help='家庭编码 (如 0622)，可多次指定')
    parser.add_argument('--codes', type=str, default='0622',
                        help='用逗号分隔的编码列表 (如 0622,0701)')
    parser.add_argument('--skip-vlm', action='store_true',
                        help='跳过 VLM API 调用，仅对已有输出评分')
    parser.add_argument('--image-dir', type=str, default='input/photo',
                        help='户型图目录 (room_{code}.png)')
    parser.add_argument('--traj-dir', type=str, default='input/traj',
                        help='轨迹数据目录')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='保存对比结果到文件')
    parser.add_argument('--excel', action='store_true',
                        help='同时输出 Excel 文件 (.xlsx)')

    args = parser.parse_args()

    # 解析编码列表
    if args.code:
        codes = args.code
    else:
        codes = [c.strip() for c in args.codes.split(',') if c.strip()]

    gt_dir = 'GT'
    all_results = {}

    for code in codes:
        pred_dir = f'output/{code}/yaml'
        baselines_dir = f'output/{code}/baselines'
        gt_path = os.path.join(gt_dir, f'layout_{code}.yaml')
        devices_path = os.path.join('input', 'Smart_device', f'smartDevice_{code}.yaml')
        if not os.path.exists(gt_path):
            print(f"[SKIP] GT 文件不存在: {gt_path}")
            continue

        vlm_generated = {
            'vlm_zeroshot': bool(args.skip_vlm),
            'vlm_cot': bool(args.skip_vlm),
        }

        # ── Step 1: VLM baselines ──
        if not args.skip_vlm:
            print(f"\n{'#'*60}")
            print(f"#  VLM Baselines — 家庭 {code}")
            print(f"{'#'*60}")

            image_path = os.path.join(args.image_dir, f'room_{code}.png')
            image_available = True
            if not os.path.exists(image_path):
                print(f"  [WARN] 图片不存在: {image_path}")
                alt_paths = [
                    os.path.join(args.image_dir, f'room_real{code[-2:]}.png'),
                    os.path.join(args.image_dir, f'room_{code[-2:]}.png'),
                ]
                for alt in alt_paths:
                    if os.path.exists(alt):
                        image_path = alt
                        print(f"  [OK] 使用图片: {image_path}")
                        break
                else:
                    print(f"  [SKIP] 找不到图片，跳过 VLM")
                    image_available = False

            devices_available = os.path.exists(devices_path)
            if not devices_available:
                print(f"  [SKIP] 智能设备输入不存在: {devices_path}")

            traj_path = os.path.join(args.traj_dir, code,
                                     f'{code}-Engineer-all_trajectory.json')
            if not os.path.exists(traj_path):
                traj_dir = os.path.join(args.traj_dir, code)
                if os.path.isdir(traj_dir):
                    jsons = sorted(Path(traj_dir).glob('*.json'),
                                   key=lambda p: p.stat().st_size, reverse=True)
                    if jsons:
                        traj_path = str(jsons[0])

            if image_available and devices_available:
                os.makedirs(baselines_dir, exist_ok=True)
                staging_parent = os.path.abspath(baselines_dir)
                with tempfile.TemporaryDirectory(
                        prefix='.vlm_run_', dir=staging_parent) as staging_dir:
                    print(f"\n  [1/2] VLM Zero-shot...")
                    cmd_zs = [
                        sys.executable, 'baselines/vlm_baseline.py',
                        '--image', image_path,
                        '--devices', devices_path,
                        '--output-dir', staging_dir,
                        '--method', 'zeroshot',
                        '--code', code,
                    ]
                    if traj_path and os.path.exists(traj_path):
                        cmd_zs += ['--trajectory', traj_path]
                    vlm_generated['vlm_zeroshot'] = generate_vlm_prediction(
                        cmd_zs,
                        os.path.join(staging_dir, f'vlm_zeroshot_{code}.yaml'),
                        os.path.join(baselines_dir, f'vlm_zeroshot_{code}.yaml'))

                    print(f"\n  [2/2] VLM Chain-of-Thought...")
                    cmd_cot = [
                        sys.executable, 'baselines/vlm_baseline.py',
                        '--image', image_path,
                        '--devices', devices_path,
                        '--output-dir', staging_dir,
                        '--method', 'cot',
                        '--code', code,
                    ]
                    if traj_path and os.path.exists(traj_path):
                        cmd_cot += ['--trajectory', traj_path]
                    vlm_generated['vlm_cot'] = generate_vlm_prediction(
                        cmd_cot,
                        os.path.join(staging_dir, f'vlm_cot_{code}.yaml'),
                        os.path.join(baselines_dir, f'vlm_cot_{code}.yaml'))

        # ── Step 2: 评估所有方法 ──
        print(f"\n{'─'*55}")
        print(f"  评估所有方法 — 家庭 {code}")
        code_results = {}

        # Ours
        ours_pred = os.path.join(pred_dir, f'03_final_{code}.yaml')
        ra, rb = run_evaluation(ours_pred, gt_path, f'Ours ({code})')
        code_results['ours_global'] = ra
        code_results['ours_aligned'] = rb

        # VLM Zero-shot
        vlm_zs_pred = os.path.join(baselines_dir, f'vlm_zeroshot_{code}.yaml')
        ra, rb = evaluate_vlm_prediction(
            vlm_generated['vlm_zeroshot'], vlm_zs_pred, gt_path,
            f'VLM Zero-shot ({code})')
        code_results['vlm_zeroshot_global'] = ra
        code_results['vlm_zeroshot_aligned'] = rb

        # VLM CoT
        vlm_cot_pred = os.path.join(baselines_dir, f'vlm_cot_{code}.yaml')
        ra, rb = evaluate_vlm_prediction(
            vlm_generated['vlm_cot'], vlm_cot_pred, gt_path,
            f'VLM CoT ({code})')
        code_results['vlm_cot_global'] = ra
        code_results['vlm_cot_aligned'] = rb

        # CV-Only (生成 + 评估)
        cv_only_pred = os.path.join(baselines_dir, f'cv_only_{code}.yaml')
        if not os.path.exists(cv_only_pred):
            print(f"\n  [生成 CV-Only...]")
            subprocess.run([
                sys.executable, 'baselines/cv_only_baseline.py',
                '--code', code,
                '--output-dir', baselines_dir,
            ], cwd=PROJECT_ROOT)
        ra, rb = run_evaluation(cv_only_pred, gt_path, f'CV-Only ({code})',
                                quiet=True)
        code_results['cv_only_global'] = ra
        code_results['cv_only_aligned'] = rb

        all_results[code] = code_results

    # ── 汇总输出 ──

    completed_codes = [code for code in codes if code in all_results]
    if not completed_codes:
        print("[ERROR] 没有可汇总的家庭结果")
        return 1

    if len(completed_codes) == 1:
        code = completed_codes[0]
        code_results = all_results[code]
        print_comparison_table(code_results)
        save_comparison_table(code_results,
                              os.path.join(PROJECT_ROOT, 'output', code, 'evaluation', f'comparison_{code}.txt'))
        if args.excel:
            save_comparison_excel(all_results, completed_codes,
                                  os.path.join(PROJECT_ROOT, 'output', code, 'evaluation', f'comparison_{code}.xlsx'))
    else:
        # 批量汇总：构建文本并同时打印和保存
        summary_lines = []
        summary_lines.append(f"{'='*80}")
        summary_lines.append(f"  对比实验汇总 — {len(completed_codes)} 个家庭")
        summary_lines.append(f"{'='*80}")

        for protocol_key, protocol_label in PROTOCOLS_ORDER:
            summary_lines.append(f"\n  Protocol: {protocol_label}")
            for method_key, method_label in METHODS_ORDER:
                vals = []
                result_key = f'{method_key}_{protocol_key}'
                for code in completed_codes:
                    r = all_results.get(code, {}).get(result_key)
                    if r and r.cls is not None:
                        vals.append(r)
                if vals:
                    avg_room_f1 = sum(r.room_f1 for r in vals) / len(vals)
                    avg_sem_f1 = sum(r.semantic_f1 for r in vals) / len(vals)
                    avg_fna = sum(r.furniture_naming_accuracy for r in vals) / len(vals)
                    avg_f1 = sum(r.furniture_f1 for r in vals) / len(vals)
                    line = (f"  {method_label:<28} RoomF1={avg_room_f1:.1%} "
                            f"LocF1={avg_f1:.1%} SemF1={avg_sem_f1:.1%} "
                            f"CondFNA={avg_fna:.1%}")
                else:
                    line = (f"  {method_label:<28} RoomF1=N/A LocF1=N/A "
                            f"SemF1=N/A CondFNA=N/A")
                summary_lines.append(line)

        summary_lines.append(
            "\n  Predictions are frozen before GT scoring; compare methods only within the same protocol.")

        for line in summary_lines:
            print(line)

        # 保存汇总结果
        eval_dir = os.path.join(PROJECT_ROOT, 'output', 'evaluation')
        save_path = os.path.join(eval_dir, 'comparison_summary.txt')
        os.makedirs(eval_dir, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        print(f"\n  汇总结果已保存至: {save_path}")

        if args.excel:
            save_comparison_excel(all_results, completed_codes,
                                  os.path.join(eval_dir, 'comparison_summary.xlsx'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
