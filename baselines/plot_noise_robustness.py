"""
实验组 B v2：噪声鲁棒性 — 可视化脚本

从 noise_robustness.py 输出的 JSON 数据读取结果，生成:

  1. B1: 设备坐标扰动 — CLS/RTA/F1 随扰动等级变化 + Retention
  2. B2: 行为轨迹稀疏度 — CLS 稳定率曲线 (含 error bar)
  3. B3: 图像退化 — F1 下降幅度柱状图

用法:
  python baselines/plot_noise_robustness.py --code 0622
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os, sys, argparse, json, glob

# ── CJK 字体设置 ──
_has_cjk_font = False
for _fname in ['Microsoft YaHei', 'SimHei', 'SimSun', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']:
    try:
        matplotlib.font_manager.findfont(_fname, fallback_to_default=False)
        plt.rcParams['font.family'] = _fname
        _has_cjk_font = True
        break
    except Exception:
        continue
if not _has_cjk_font:
    plt.rcParams['font.family'] = 'sans-serif'

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# 图 1: B1 设备坐标扰动 — Correlated Perturbation
# ============================================================

def plot_b1_device_perturbation(data: dict, output_dir: str, code: str):
    """B1 v2: 设备坐标扰动 — 5 条指标曲线 + Retention"""
    if not data:
        print("[警告] B1 数据为空，跳过画图")
        return

    items = sorted(
        [(k, v) for k, v in data.items() if v is not None],
        key=lambda x: x[1].get('level', int(x[0]))
    )

    if len(items) < 2:
        print("[警告] B1 数据不足(<2点)，跳过画图")
        return

    levels = [v.get('level', int(k)) for k, v in items]
    cls_vals = [v.get('cls', 0) * 100 for _, v in items]
    rta_vals = [v.get('rta', 0) * 100 for _, v in items]
    f1_vals = [v.get('f1', 0) * 100 for _, v in items]
    fna_vals = [v.get('fna', 0) * 100 for _, v in items]
    cf1_vals = [v.get('centroid_f1', 0) * 100 for _, v in items]

    # Retention
    cls0 = cls_vals[0] if cls_vals else 100.0
    retention = [c / cls0 * 100 if cls0 > 0 else 100 for c in cls_vals]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(levels, cls_vals, 'o-', color='#2E86C1', linewidth=2.5, markersize=9,
            label='CLS', markerfacecolor='white', markeredgewidth=2)
    ax.plot(levels, rta_vals, 's--', color='#E74C3C', linewidth=2, markersize=9,
            label='RTA', markerfacecolor='white', markeredgewidth=2)
    ax.plot(levels, f1_vals, 'D-.', color='#27AE60', linewidth=2, markersize=8,
            label='F1@IoU', markerfacecolor='white', markeredgewidth=2)
    ax.plot(levels, fna_vals, '^:', color='#F39C12', linewidth=2, markersize=8,
            label='FNA', markerfacecolor='white', markeredgewidth=2)
    ax.plot(levels, cf1_vals, 'v--', color='#8E44AD', linewidth=2, markersize=8,
            label='CentroidF1', markerfacecolor='white', markeredgewidth=2)

    # 10dm threshold
    ax.axvline(x=10, color='#7F8C8D', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.annotate('10dm threshold', xy=(10, ax.get_ylim()[1] * 0.92),
                xytext=(11.5, ax.get_ylim()[1] * 0.95),
                fontsize=9, color='#7F8C8D',
                arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=1.2))

    # 数值标注 + Retention
    for i, lvl in enumerate(levels):
        ax.annotate(f'{cls_vals[i]:.1f}', (lvl, cls_vals[i]),
                    textcoords="offset points", xytext=(0, 12),
                    ha='center', fontsize=7.5, color='#2E86C1', fontweight='bold')
        ax.annotate(f'{rta_vals[i]:.1f}', (lvl, rta_vals[i]),
                    textcoords="offset points", xytext=(0, -16),
                    ha='center', fontsize=7.5, color='#E74C3C')
        ax.annotate(f'{f1_vals[i]:.1f}', (lvl, f1_vals[i]),
                    textcoords="offset points", xytext=(12, 6),
                    ha='center', fontsize=7.5, color='#27AE60')
        ax.annotate(f'{fna_vals[i]:.1f}', (lvl, fna_vals[i]),
                    textcoords="offset points", xytext=(-12, 6),
                    ha='center', fontsize=7.5, color='#F39C12')
        ax.annotate(f'{cf1_vals[i]:.1f}', (lvl, cf1_vals[i]),
                    textcoords="offset points", xytext=(12, -10),
                    ha='center', fontsize=7.5, color='#8E44AD')

    ax.set_xlabel('Perturbation Level (+/-dm)', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('B1: Device Coordinate Perturbation (Correlated)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower left', ncol=5, framealpha=0.9,
              edgecolor='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_xticks(levels)

    # 添加 Retention 文本
    ret_text = "CLS Retention: " + ", ".join(
        f"@{l}dm={retention[i]:.0f}%" for i, l in enumerate(levels) if l > 0)
    ax.text(0.98, 0.02, ret_text, transform=ax.transAxes, fontsize=8,
            ha='right', va='bottom', color='#7F8C8D',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8F9FA', edgecolor='#cccccc', alpha=0.8))

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    for fmt in ['pdf', 'png']:
        path = os.path.join(output_dir, f'noise_b1_device_perturbation_{code}.{fmt}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')
    plt.close()


# ============================================================
# 图 2: B2 轨迹稀疏度 — 含 error bar
# ============================================================

def plot_b2_trajectory_sparsity(data: dict, output_dir: str, code: str):
    """B2 v2: 轨迹稀疏度 — temporal downsampling, multi-seed error bars"""
    if not data:
        print("[警告] B2 数据为空，跳过画图")
        return

    items = sorted(
        [(k, v) for k, v in data.items() if v is not None],
        key=lambda x: x[1].get('step', 1)
    )

    if len(items) < 2:
        print("[警告] B2 数据不足(<2点)，跳过画图")
        return

    labels = [k for k, _ in items]
    steps = [v.get('step', 1) for _, v in items]

    def _get(v, attr):
        if v.get('multi_seed'):
            std_key = f'{attr}_std'
            return v.get(attr, 0) * 100, v.get(std_key, 0) * 100
        return v.get(attr, 0) * 100, 0.0

    cls_v = [_get(v, 'cls') for _, v in items]
    f1_v = [_get(v, 'f1') for _, v in items]
    rta_v = [_get(v, 'rta') for _, v in items]
    cf1_v = [_get(v, 'centroid_f1') for _, v in items]

    cls_means = [c[0] for c in cls_v]
    cls_errs = [c[1] for c in cls_v]
    f1_means = [c[0] for c in f1_v]
    f1_errs = [c[1] for c in f1_v]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── 左图: CLS with error bars ──
    ax = axes[0]
    x = np.arange(len(labels))
    colors = ['#27AE60', '#F39C12', '#E74C3C', '#8E44AD'][:len(labels)]
    bars = ax.bar(x, cls_means, yerr=cls_errs, color=colors, width=0.5,
                  edgecolor='white', linewidth=0.8, capsize=5, error_kw={'linewidth': 1.2})
    for bar, val, err in zip(bars, cls_means, cls_errs):
        label = f'{val:.1f}%' + (f' +/-{err:.1f}%' if err > 0.5 else '')
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 1.5,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold')

    full_cls = cls_means[0] if cls_means else 100
    for bar, val in zip(bars, cls_means):
        stability = val / full_cls * 100 if full_cls > 0 else 100
        if stability < 95:
            ax.text(bar.get_x() + bar.get_width() / 2, val / 2,
                    f'{stability:.0f}%', ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('CLS (%)', fontsize=12)
    ax.set_title('CLS vs Temporal Sparsity', fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # ── 右图: F1 + RTA + CentroidF1 curves ──
    ax2 = axes[1]
    ax2.errorbar(x, f1_means, yerr=f1_errs, fmt='D-', color='#27AE60',
                 linewidth=2, markersize=8, label='F1@IoU', capsize=5,
                 markerfacecolor='white', markeredgewidth=2)
    rta_means = [c[0] for c in rta_v]
    rta_errs = [c[1] for c in rta_v]
    ax2.errorbar(x, rta_means, yerr=rta_errs, fmt='s--', color='#E74C3C',
                 linewidth=2, markersize=8, label='RTA', capsize=5,
                 markerfacecolor='white', markeredgewidth=2)
    cf1_means = [c[0] for c in cf1_v]
    cf1_errs = [c[1] for c in cf1_v]
    ax2.errorbar(x, cf1_means, yerr=cf1_errs, fmt='^:', color='#8E44AD',
                 linewidth=2, markersize=8, label='CentroidF1', capsize=5,
                 markerfacecolor='white', markeredgewidth=2)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel('Score (%)', fontsize=12)
    ax2.set_title('Detection Metrics vs Sparsity', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    fig.suptitle('B2: Trajectory Temporal Sparsity (multi-seed)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    for fmt in ['pdf', 'png']:
        path = os.path.join(output_dir, f'noise_b2_trajectory_sparsity_{code}.{fmt}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')
    plt.close()


# ============================================================
# 图 3: B3 图像退化 — realistic corruption
# ============================================================

def plot_b3_image_degradation(data: dict, output_dir: str, code: str):
    """B3 v2: 图像退化 — realistic corruption (JPEG, Gamma, Noise, Blur)"""
    if not data:
        print("[警告] B3 数据为空，跳过画图")
        return

    variant_order = [
        "Clean (No Degradation)",
        "Rescale 0.5x",
        "Rescale 0.25x",
        "Map Crop 20%",
        "SLAM Wall Noise (sigma=1.5)",
        "SLAM Wall Noise (sigma=3.0)",
    ]
    ordered = [(k, data[k]) for k in variant_order if k in data and data[k] is not None]

    if len(ordered) < 2:
        print("[警告] B3 数据不足(<2点)，跳过画图")
        return

    names = [k.replace(' (No Degradation)', '\n(Clean)')
              .replace('Rescale 0.5x', 'Rescale\n0.5x')
              .replace('Rescale 0.25x', 'Rescale\n0.25x')
              .replace('Map Crop 20%', 'Map Crop\n20%')
              .replace('SLAM Wall Noise (sigma=1.5)', 'WallNoise\n(s=1.5)')
              .replace('SLAM Wall Noise (sigma=3.0)', 'WallNoise\n(s=3.0)')
              for k, _ in ordered]

    f1_vals = [v.get('f1', 0) * 100 for _, v in ordered]
    fna_vals = [v.get('fna', 0) * 100 for _, v in ordered]
    cls_vals = [v.get('cls', 0) * 100 for _, v in ordered]
    centroid_recalls = [v.get('centroid_recall', 0) * 100 for _, v in ordered]
    small_recalls = []
    for _, v in ordered:
        pc = v.get('per_class', {})
        small_pc = pc.get('small (<50)', {})
        sr = small_pc.get('recall', 0) * 100 if small_pc else 0
        small_recalls.append(sr)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── 左图: F1 + FNA + CLS ──
    ax = axes[0]
    n = len(names)
    x = np.arange(n)
    width = 0.25

    bars_f1 = ax.bar(x - width, f1_vals, width, label='F1@IoU',
                     color='#2E86C1', edgecolor='white', linewidth=0.6)
    bars_fna = ax.bar(x, fna_vals, width, label='FNA',
                      color='#F39C12', edgecolor='white', linewidth=0.6)
    bars_cls = ax.bar(x + width, cls_vals, width, label='CLS',
                      color='#27AE60', edgecolor='white', linewidth=0.6)

    for bars in [bars_f1, bars_fna, bars_cls]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('F1 / FNA / CLS under Corruption', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # ── 右图: 小家具 recall + CentroidR ──
    ax2 = axes[1]
    width2 = 0.3

    bars_sr = ax2.bar(x - width2 / 2, small_recalls, width2, label='Small Furniture Recall',
                      color='#E74C3C', edgecolor='white', linewidth=0.6)
    bars_cr = ax2.bar(x + width2 / 2, centroid_recalls, width2, label='Centroid Recall',
                      color='#8E44AD', edgecolor='white', linewidth=0.6)

    for bar, val in zip(bars_sr, small_recalls):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold',
                 color='#E74C3C')
    for bar, val in zip(bars_cr, centroid_recalls):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold',
                 color='#8E44AD')

    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=8)
    ax2.set_ylabel('Recall (%)', fontsize=12)
    ax2.set_title('Small Furniture & Centroid Recall', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='lower right')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    fig.suptitle('B3: Realistic Image Corruption Robustness',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    for fmt in ['pdf', 'png']:
        path = os.path.join(output_dir, f'noise_b3_image_degradation_{code}.{fmt}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')
    plt.close()

    # ── 额外: F1 下降 ──
    if len(f1_vals) >= 2:
        clean_f1 = f1_vals[0]
        fig2, ax3 = plt.subplots(figsize=(8, 4))
        drops = []
        drop_labels = []
        for i in range(1, len(names)):
            drop = clean_f1 - f1_vals[i]
            drops.append(drop)
            drop_labels.append(names[i].replace('\n', ' '))

        colors_drop = ['#E74C3C' if d > 0 else '#27AE60' for d in drops]
        bars_drop = ax3.barh(range(len(drops)), drops, color=colors_drop, height=0.5,
                             edgecolor='white', linewidth=0.8)
        for bar, val in zip(bars_drop, drops):
            ax3.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                     f'{val:+.1f}%', va='center', fontsize=11, fontweight='bold')

        ax3.set_yticks(range(len(drops)))
        ax3.set_yticklabels(drop_labels, fontsize=10)
        ax3.set_xlabel('F1 Drop (%)', fontsize=12)
        ax3.axvline(x=0, color='black', linewidth=0.8)
        ax3.set_title('B3: F1 Drop under Image Corruption', fontsize=13, fontweight='bold')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.grid(axis='x', alpha=0.3, linestyle='--')
        ax3.set_axisbelow(True)
        ax3.invert_yaxis()

        plt.tight_layout()
        for fmt in ['pdf', 'png']:
            path = os.path.join(output_dir, f'noise_b3_f1_drop_{code}.{fmt}')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'Saved: {path}')
        plt.close()


# ============================================================
# 汇总图
# ============================================================

def plot_summary(all_data: dict, output_dir: str, code: str):
    """汇总图: B1 CLS 曲线 + B2 稳定率 + B3 F1 下降"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── B1 ──
    ax = axes[0]
    b1_data = all_data.get('b1', {})
    if b1_data:
        items = sorted(
            [(k, v) for k, v in b1_data.items() if v is not None],
            key=lambda x: x[1].get('level', int(x[0]))
        )
        if len(items) >= 2:
            levels = [v.get('level', int(k)) for k, v in items]
            cls_v = [v.get('cls', 0) * 100 for _, v in items]
            rta_v = [v.get('rta', 0) * 100 for _, v in items]
            ax.plot(levels, cls_v, 'o-', color='#2E86C1', linewidth=2.5, markersize=8,
                    label='CLS', markerfacecolor='white', markeredgewidth=2)
            ax.plot(levels, rta_v, 's--', color='#E74C3C', linewidth=2, markersize=8,
                    label='RTA', markerfacecolor='white', markeredgewidth=2)
            ax.axvline(x=10, color='#7F8C8D', linestyle=':', linewidth=1.5, alpha=0.7)
            ax.set_xticks(levels)
            ax.legend(fontsize=8)
    ax.set_xlabel('Perturbation (+/-dm)', fontsize=10)
    ax.set_ylabel('Score (%)', fontsize=10)
    ax.set_title('B1: Correlated Perturbation', fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # ── B2 ──
    ax2 = axes[1]
    b2_data = all_data.get('b2', {})
    if b2_data:
        items = sorted(
            [(k, v) for k, v in b2_data.items() if v is not None],
            key=lambda x: x[1].get('step', 1)
        )
        if len(items) >= 2:
            labels_b2 = [k for k, _ in items]
            cls_v2 = [v.get('cls', 0) * 100 for _, v in items]
            cls_err = [v.get('cls_std', 0) * 100 for _, v in items]
            full_cls = cls_v2[0] if cls_v2 else 100
            colors = ['#27AE60', '#F39C12', '#E74C3C', '#8E44AD'][:len(items)]
            bars = ax2.bar(range(len(items)), cls_v2, yerr=cls_err,
                           color=colors, width=0.5, edgecolor='white', capsize=4)
            for bar, val in zip(bars, cls_v2):
                stability = val / full_cls * 100 if full_cls > 0 else 100
                ax2.text(bar.get_x() + bar.get_width() / 2, val + 1,
                         f'{val:.1f}%\n({stability:.0f}%)', ha='center', fontsize=8)
            ax2.set_xticks(range(len(items)))
            ax2.set_xticklabels(labels_b2, fontsize=8)
    ax2.set_ylabel('CLS (%)', fontsize=10)
    ax2.set_title('B2: Temporal Sparsity', fontsize=11, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # ── B3 ──
    ax3 = axes[2]
    b3_data = all_data.get('b3', {})
    if b3_data:
        variant_order = [
            "Clean (No Degradation)",
            "Rescale 0.5x",
            "Rescale 0.25x",
            "Map Crop 20%",
            "SLAM Wall Noise (sigma=1.5)",
            "SLAM Wall Noise (sigma=3.0)",
        ]
        ordered = [(k, b3_data[k]) for k in variant_order if k in b3_data and b3_data[k] is not None]
        if len(ordered) >= 2:
            clean_f1 = ordered[0][1].get('f1', 0) * 100
            names_b3 = [k.replace(' (No Degradation)', '').replace('Rescale 0.5x', 'Rescale0.5x')
                         .replace('Rescale 0.25x', 'Rescale0.25x')
                         .replace('Map Crop 20%', 'MapCrop20%')
                         .replace('SLAM Wall Noise (sigma=1.5)', 'WallNoise(1.5)')
                         .replace('SLAM Wall Noise (sigma=3.0)', 'WallNoise(3.0)') for k, _ in ordered[1:]]
            drops = [clean_f1 - v.get('f1', 0) * 100 for _, v in ordered[1:]]
            colors_d = ['#E74C3C' if d > 0 else '#27AE60' for d in drops]
            ax3.barh(range(len(drops)), drops, color=colors_d, height=0.5,
                     edgecolor='white')
            for i, d in enumerate(drops):
                ax3.text(d + 0.3, i, f'{d:+.1f}%', va='center', fontsize=10, fontweight='bold')
            ax3.set_yticks(range(len(drops)))
            ax3.set_yticklabels(names_b3, fontsize=9)
            ax3.axvline(x=0, color='black', linewidth=0.8)
            ax3.invert_yaxis()
    ax3.set_xlabel('F1 Drop (%)', fontsize=10)
    ax3.set_title('B3: F1 Drop', fontsize=11, fontweight='bold')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(axis='x', alpha=0.3, linestyle='--')

    fig.suptitle('Experiment Group B v2: Noise Robustness Summary',
                 fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    for fmt in ['pdf', 'png']:
        path = os.path.join(output_dir, f'noise_robustness_summary_{code}.{fmt}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')
    plt.close()


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='实验组 B v2 噪声鲁棒性 — 可视化')
    parser.add_argument('--code', default='0622', help='家庭编码 (默认: 0622)')
    parser.add_argument('--input', '-i', default=None,
                        help='结构化 JSON 结果文件 (默认自动查找最新)')
    args = parser.parse_args()

    code = args.code

    if args.input:
        json_path = args.input
    else:
        eval_dir = os.path.join(PROJECT_ROOT, 'output', code, 'evaluation')
        candidates = sorted(
            glob.glob(os.path.join(eval_dir, 'noise_robustness_*.json')),
            key=os.path.getmtime, reverse=True
        )
        if not candidates:
            print(f"[错误] 未找到 JSON 结果文件: {eval_dir}/noise_robustness_*.json")
            print(f"  请先运行: python baselines/noise_robustness.py --code {code}")
            sys.exit(1)
        json_path = candidates[0]
        print(f"自动选择最新结果: {os.path.basename(json_path)}")

    with open(json_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    output_dir = os.path.join(PROJECT_ROOT, 'output', code, 'figures')

    if all_data.get('b1'):
        plot_b1_device_perturbation(all_data['b1'], output_dir, code)
    if all_data.get('b2'):
        plot_b2_trajectory_sparsity(all_data['b2'], output_dir, code)
    if all_data.get('b3'):
        plot_b3_image_degradation(all_data['b3'], output_dir, code)
    if all_data:
        plot_summary(all_data, output_dir, code)

    print('\nDone. All figures saved.')


if __name__ == '__main__':
    main()
