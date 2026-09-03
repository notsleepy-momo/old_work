"""
论文对比实验画图脚本

从 compare_baselines.py 输出的 comparison_{code}.txt 自动读取数据，
分别生成 Global 与 Aligned 两套统一协议的分组柱状图 (Room F1,
Localization F1, Semantic F1, Conditional FNA)。

用法:
  # 自动查找 output/{code}/evaluation/comparison_{code}.txt
  python baselines/plot_comparison.py --code 0622

  # 指定输入文件
  python baselines/plot_comparison.py --code 0622 --input output/comparison.txt
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, sys, argparse, re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def parse_comparison_file(filepath: str) -> dict:
    """解析两个协议，返回 {协议: {方法: 四个百分比指标}}。"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"对比结果文件不存在: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    protocol_pattern = re.compile(
        r'^\s*Protocol:\s+[AB]\.\s+.*\((Global|Aligned)\)\s*$')
    row_pattern = re.compile(
        r'^\s{2}([\w ()+\-]+?)\s{2,}'
        r'([\d.]+)%\s+'                     # Room F1
        r'([\d.]+)%\s+'                     # Localization F1
        r'([\d.]+)%\s+'                     # Semantic F1
        r'([\d.]+)%\s+'                     # Conditional FNA
        r'(\d+)\s*$'                         # Semantic TP
    )

    # 方法名映射 (comparison.txt 中的名称 -> plot 中的简写)
    name_map = {
        'VLM Zero-shot':           'Zeroshot',
        'VLM Chain-of-Thought':    'CoT',
        'CV-Only (No LLM)':        'CV-Only',
        'Ours':                    'Ours',
    }

    results = {'Global': {}, 'Aligned': {}}
    current_protocol = None
    for line in content.splitlines():
        protocol_match = protocol_pattern.match(line)
        if protocol_match:
            current_protocol = protocol_match.group(1)
            continue
        if current_protocol is None:
            continue
        row_match = row_pattern.match(line)
        if not row_match:
            continue
        raw_name = row_match.group(1).strip()
        values = [float(row_match.group(i)) for i in range(2, 6)]
        label = name_map.get(raw_name, raw_name)
        results[current_protocol][label] = values

    return results


def plot_protocol(data_dict: dict, protocol: str, output_dir: str):
    """为单个协议绘制并保存四项指标。"""
    method_order = ['Zeroshot', 'CoT', 'CV-Only', 'Ours']
    method_labels = ['VLM\nZero-shot', 'VLM\nCoT', 'CV-Only\n(No LLM)', 'Ours']
    metrics = ['Room F1', 'Localization F1', 'Semantic F1', 'Conditional FNA']

    data_rows = []
    valid_labels = []
    for key, label in zip(method_order, method_labels):
        if key in data_dict:
            data_rows.append(data_dict[key])
            valid_labels.append(label)

    if not data_rows:
        print(f"  [跳过] {protocol} 协议没有可绘制的数据")
        return False

    data = np.array(data_rows)
    print(f"  {protocol}: 解析到 {len(data_rows)} 个方法的数据")

    colors = ['#E8C170', '#D4A030', '#7FB5B5', '#4A90D9'][:len(data_rows)]
    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(metrics))
    width = 0.15
    n = len(data_rows)
    for i in range(n):
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, data[i], width, label=valid_labels[i],
                      color=colors[i], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, data[i]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=7)

    ax.set_title(f'{protocol} Matching Protocol', fontsize=13, loc='left', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_ylim(0, 115)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20),
              fontsize=9, ncol=min(n, 4), framealpha=0.9, edgecolor='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    plt.tight_layout()

    stem = f'comparison_baselines_{protocol.lower()}'
    for fmt in ['pdf', 'png']:
        path = os.path.join(output_dir, f'{stem}.{fmt}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description='对比实验柱状图')
    parser.add_argument('--code', default='0622',
                        help='家庭编码 (默认: 0622)')
    parser.add_argument('--input', '-i', default=None,
                        help='对比结果文件 (默认: output/{code}/evaluation/comparison_{code}.txt)')
    args = parser.parse_args()

    code = args.code
    input_file = args.input or os.path.join(
        PROJECT_ROOT, 'output', code, 'evaluation', f'comparison_{code}.txt')

    # ── 解析数据 ──
    try:
        data_dict = parse_comparison_file(input_file)
    except FileNotFoundError as e:
        print(f"[错误] {e}")
        sys.exit(1)

    if not any(data_dict.values()):
        print(f"[错误] 未能从 {input_file} 解析到任何数据")
        sys.exit(1)

    output_dir = os.path.join(PROJECT_ROOT, 'output', code, 'figures')
    os.makedirs(output_dir, exist_ok=True)
    for protocol in ('Global', 'Aligned'):
        plot_protocol(data_dict[protocol], protocol, output_dir)
    print('Done.')


if __name__ == '__main__':
    main()
