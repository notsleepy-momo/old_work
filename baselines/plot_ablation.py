"""
消融实验画图脚本

从 ablation_experiments.py 输出的 ablation_results_*.txt 读取数据，
生成:
  1. 模态贡献消融柱状图 — 展示各模态独立与叠加贡献
  2. Constrained vs Free-form LLM 对比图 — 展示协议违规率降低效果

用法:
  # 指定消融结果文件
  python baselines/plot_ablation.py --code 0622

  # 或直接指定输入文件
  python baselines/plot_ablation.py --input output/0622/evaluation/ablation_results_20250101_120000.txt
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, sys, argparse, re, glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# 数据解析
# ============================================================

def parse_ablation_file(filepath: str) -> dict:
    """逐行解析消融结果文件，返回结构化数据。

    通过识别节标题 ("Modal Contribution" / "Constrained vs Free-form LLM")
    和数据行 (以 2 空格 + 非空格开头) 来提取表格数据。
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"消融结果文件不存在: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    result = {'modal': None, 'llm': None}

    # ── 识别节边界 ──
    section_starts = {}  # section_name → line_index
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('Modal Contribution'):
            section_starts['modal'] = i
        elif stripped.startswith('Constrained vs Free-form LLM'):
            section_starts['llm'] = i

    # ── 通用数据行解析: "  Name              12.3%  45.6%  ..." ──
    def _parse_data_rows(start_idx: int, expected_cols: int = 5) -> list:
        """从 start_idx 开始，查找表头后的数据行，返回 [{name, vals}]

        停止条件: 遇到分隔线 (全由 '-' 组成) 且已收集到数据。
        """
        rows = []
        in_header = False
        for i in range(start_idx, min(start_idx + 30, len(lines))):
            line = lines[i]
            # 检测表头行 (包含 "Variant" 关键词)
            if not in_header and 'Variant' in line:
                in_header = True
                continue
            if not in_header:
                continue
            # 遇到分隔线 → 如果已有数据则停止，否则跳过
            stripped = line.strip()
            if stripped and set(stripped) <= {'-'}:
                if rows:
                    break
                continue
            # 检测数据行: 2+ 空格开头 + 非空格字符，且包含 ≥ expected_cols 个百分比值
            if re.match(r'^\s{2,}\S', line):
                pcts = re.findall(r'([\d.]+)%', line)
                if len(pcts) >= expected_cols:
                    name = line.strip().split('%')[0].split('  ')[0].strip()
                    name = re.sub(r'\s+\d+/\d+$', '', name).strip()
                    vals = [float(v) for v in pcts[:expected_cols]]
                    rows.append({'name': name, 'vals': vals})
        return rows

    # ── 解析 Modal ──
    if 'modal' in section_starts:
        rows = _parse_data_rows(section_starts['modal'], expected_cols=5)
        if rows:
            result['modal'] = {
                'variants': [r['name'] for r in rows],
                'RTA':  [r['vals'][0] for r in rows],
                'F1':   [r['vals'][1] for r in rows],
                'FNA':  [r['vals'][2] for r in rows],
                'CLS':  [r['vals'][3] for r in rows],
                'Hallu':[r['vals'][4] for r in rows],
            }
            label_map = {
                'Layout Only': 'Layout\nOnly',
                '+ Device': '+Device',
                '+ Trajectory': '+Traj.',
                'Full (All Modalities)': 'Full',
            }
            result['modal']['labels'] = [
                label_map.get(v, v) for v in result['modal']['variants']]

    # ── 解析 LLM ──
    if 'llm' in section_starts:
        rows = _parse_data_rows(section_starts['llm'], expected_cols=6)
        if rows:
            result['llm'] = {
                'variants': [r['name'] for r in rows],
                # Keep the old internal key for plotting compatibility.
                'Hallu': [r['vals'][0] for r in rows],  # Protocol violation rate
                'SemFDR':[r['vals'][1] for r in rows],
                'RTA':   [r['vals'][2] for r in rows],
                'F1':    [r['vals'][3] for r in rows],
                'FNA':   [r['vals'][4] for r in rows],
                'CLS':   [r['vals'][5] for r in rows],
            }
            llm_label_map = {
                'Free-form LLM': 'Free-form',
                'Constrained (Ours)': 'Constrained\n(Ours)',
            }
            result['llm']['labels'] = [
                llm_label_map.get(v, v) for v in result['llm']['variants']]

            # 提取协议违规率降幅
            for i in range(section_starts['llm'],
                           min(section_starts['llm'] + 30, len(lines))):
                m = re.search(r'Protocol-violation reduction:\s*([\d.]+)%', lines[i])
                if m:
                    result['llm']['hallu_reduction'] = float(m.group(1))
                    break

    return result


# ============================================================
# 图 1: 模态贡献消融柱状图
# ============================================================

def plot_modal_contribution(data: dict, output_dir: str, code: str):
    """模态贡献消融 — 累加式柱状图展示各模态独立与叠加贡献"""
    variants = data['variants']
    labels = data['labels']
    metrics = ['RTA', 'F1@IoU', 'FNA', 'CLS']
    metric_keys = ['RTA', 'F1', 'FNA', 'CLS']

    # 构建数据矩阵
    mat = np.array([[data[k][i] for k in metric_keys] for i in range(len(variants))])

    n = len(variants)

    # 配色: 从浅到深 (Layout Only → Full)，动态适配变体数
    base_colors = ['#D4E6F1', '#A9CCE3', '#7FB3D8', '#5499C7', '#2E86C1', '#1A5276']
    colors = base_colors[:n] if n <= len(base_colors) else (base_colors * (n // len(base_colors) + 1))[:n]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(metrics))
    width = 0.18
    n = len(variants)

    for i in range(n):
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, mat[i], width, label=labels[i],
                      color=colors[i], edgecolor='white', linewidth=0.6)
        for bar, val in zip(bars, mat[i]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=7.5,
                    fontweight='bold' if i == n - 1 else 'normal')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_ylim(0, 110)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # 标题 (在上方留够空间给图例)
    ax.set_title('Ablation: Modal Contribution', fontsize=13, fontweight='bold', pad=28)

    # 图例放在图下方，不遮挡柱体
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
              fontsize=9, ncol=min(n, 4), framealpha=0.9, edgecolor='#cccccc')

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    for fmt in ['pdf', 'png']:
        path = os.path.join(output_dir, f'ablation_modal_{code}.{fmt}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')
    plt.close()


# ============================================================
# 图 2: Constrained vs Free-form LLM 对比图
# ============================================================

def plot_llm_constrained(data: dict, output_dir: str, code: str):
    """Constrained vs Free-form LLM — 双图对比: Protocol Violation + FNA"""
    variants = data['variants']
    labels = data['labels']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # ── 左图: Protocol Violation Rate (↓ 越低越好) ──
    ax = axes[0]
    hallu_vals = data['Hallu']
    bar_colors = ['#E74C3C', '#27AE60']
    bars = ax.bar(labels, hallu_vals, color=bar_colors, width=0.45, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, hallu_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel('Protocol Violation Rate (%)', fontsize=11)
    ax.set_ylim(0, max(hallu_vals) * 1.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # 标注协议违规率降幅
    reduction = data.get('hallu_reduction', None)
    if reduction is not None and len(hallu_vals) == 2:
        ax.annotate(f'↓ {reduction:.0f}%',
                    xy=(0.5, 1.02), xycoords='axes fraction',
                    ha='center', fontsize=10, fontweight='bold',
                    color='#27AE60',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#D5F5E3',
                              edgecolor='#27AE60', alpha=0.8))

    # ── 右图: FNA + CLS (↑ 越高越好) ──
    ax2 = axes[1]
    x = np.arange(2)
    width = 0.3
    fna_vals = data['FNA']
    cls_vals = data['CLS']

    bars_fna = ax2.bar(x - width / 2, fna_vals, width, label='FNA',
                       color='#3498DB', edgecolor='white', linewidth=0.8)
    bars_cls = ax2.bar(x + width / 2, cls_vals, width, label='CLS',
                       color='#F39C12', edgecolor='white', linewidth=0.8)

    for bar, val in zip(bars_fna, fna_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, val in zip(bars_cls, cls_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel('Score (%)', fontsize=11)
    ax2.set_ylim(0, 110)
    ax2.legend(fontsize=9, loc='lower right')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    fig.suptitle('Ablation: Constrained vs Free-form LLM', fontsize=13, fontweight='bold', y=1.06)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    os.makedirs(output_dir, exist_ok=True)
    for fmt in ['pdf', 'png']:
        path = os.path.join(output_dir, f'ablation_llm_{code}.{fmt}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')
    plt.close()


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='消融实验画图')
    parser.add_argument('--code', default='0622',
                        help='家庭编码 (默认: 0622)')
    parser.add_argument('--input', '-i', default=None,
                        help='消融结果文件 (默认自动查找最新的)')
    args = parser.parse_args()

    code = args.code

    # 自动查找最新的消融结果文件
    if args.input:
        input_file = args.input
    else:
        eval_dir = os.path.join(PROJECT_ROOT, 'output', code, 'evaluation')
        candidates = sorted(
            glob.glob(os.path.join(eval_dir, 'ablation_results_*.txt')),
            key=os.path.getmtime, reverse=True
        )
        if not candidates:
            print(f"[错误] 未找到消融结果文件: {eval_dir}/ablation_results_*.txt")
            print(f"  请先运行: python baselines/ablation_experiments.py --code {code}")
            sys.exit(1)
        input_file = candidates[0]
        print(f"自动选择最新结果: {os.path.basename(input_file)}")

    # ── 解析 ──
    try:
        data = parse_ablation_file(input_file)
    except FileNotFoundError as e:
        print(f"[错误] {e}")
        sys.exit(1)

    output_dir = os.path.join(PROJECT_ROOT, 'output', code, 'figures')

    # ── 画图 ──
    if data.get('modal') and len(data['modal'].get('variants', [])) >= 2:
        plot_modal_contribution(data['modal'], output_dir, code)
    else:
        print("[警告] 未找到 Modal Contribution 数据，跳过")

    if data.get('llm') and len(data['llm'].get('variants', [])) >= 2:
        plot_llm_constrained(data['llm'], output_dir, code)
    else:
        print("[警告] 未找到 Constrained vs Free-form LLM 数据，跳过")

    print('Done.')


if __name__ == '__main__':
    main()
