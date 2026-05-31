"""
论文配图生成脚本
生成用于作品说明书的数据分析图表（PNG格式）
用法: python gen_figures.py --output ./output --fig ./output/figures
"""

import os
import sys
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

# 设置中文字体
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 150


def fig1_source_stats(output_dir, fig_dir):
    """图1: 各源数据规模对比（条形图）"""
    sources = {
        "ADS-B": (12667, 17),
        "Remote ID": (112909, 502),
        "雷达": (32263, 500),
        "频谱": (155953, 2657),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    names = list(sources.keys())
    records = [v[0] for v in sources.values()]
    targets = [v[1] for v in sources.values()]

    bars = axes[0].bar(names, records, color=colors, edgecolor="white", linewidth=0.8)
    axes[0].set_title("各源数据记录数", fontsize=13, pad=10)
    axes[0].set_ylabel("记录数（条）")
    for bar, val in zip(bars, records):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1000,
                     f"{val:,}", ha="center", va="bottom", fontsize=10)
    axes[0].set_ylim(0, max(records) * 1.15)
    axes[0].spines[["top", "right"]].set_visible(False)

    bars2 = axes[1].bar(names, targets, color=colors, edgecolor="white", linewidth=0.8)
    axes[1].set_title("各源目标数量", fontsize=13, pad=10)
    axes[1].set_ylabel("目标数（个）")
    for bar, val in zip(bars2, targets):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                     str(val), ha="center", va="bottom", fontsize=10)
    axes[1].set_ylim(0, max(targets) * 1.15)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.suptitle("图1  异构多源数据规模统计", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(fig_dir, "fig1_source_stats.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  ✓ {path}")


def fig2_classification(output_dir, fig_dir):
    """图2: 目标分类分布（饼图）"""
    cls_path = os.path.join(output_dir, "classification_results.csv")
    if not os.path.exists(cls_path):
        print("  跳过 fig2：classification_results.csv 不存在")
        return

    df = pd.read_csv(cls_path)
    counts = df["category"].value_counts()

    label_map = {
        "ga": "通用航空",
        "uav_consumer": "消费级无人机",
        "uav_industrial": "工业级无人机",
        "uav_unknown": "未知无人机",
        "unknown": "未知目标",
    }
    labels = [label_map.get(k, k) for k in counts.index]
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=labels,
        colors=colors[:len(labels)],
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.75,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    for t in autotexts:
        t.set_fontsize(10)
    ax.set_title("图2  目标分类分布", fontsize=14, pad=15)
    plt.tight_layout()
    path = os.path.join(fig_dir, "fig2_classification.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  ✓ {path}")


def fig3_anomaly_distribution(output_dir, fig_dir):
    """图3: 异常类型分布（横向条形图）"""
    anom_path = os.path.join(output_dir, "anomaly_events.csv")
    if not os.path.exists(anom_path):
        print("  跳过 fig3：anomaly_events.csv 不存在")
        return

    df = pd.read_csv(anom_path)
    counts = df["type"].value_counts().head(10)

    label_map = {
        "signal_loss": "信号中断",
        "stall_speed": "失速告警",
        "overspeed": "超速",
        "impossible_accel": "加速度超限",
        "excessive_turn": "转弯率超限",
        "altitude_ceiling": "高度超限",
        "statistical_deviation_heading": "航向统计偏离",
        "statistical_deviation_speed": "速度统计偏离",
        "excessive_descent": "下降率超限",
        "excessive_climb": "爬升率超限",
    }
    labels = [label_map.get(k, k) for k in counts.index]
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.9, len(counts)))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(labels[::-1], counts.values[::-1], color=colors[::-1], edgecolor="white")
    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=10)
    ax.set_xlabel("事件数量（条）")
    ax.set_title("图3  异常事件类型分布（Top 10）", fontsize=14, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, counts.max() * 1.15)
    plt.tight_layout()
    path = os.path.join(fig_dir, "fig3_anomaly_dist.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  ✓ {path}")


def fig4_conflict_levels(output_dir, fig_dir):
    """图4: 冲突预警等级分布"""
    conf_path = os.path.join(output_dir, "conflict_events.csv")
    if not os.path.exists(conf_path):
        print("  跳过 fig4：conflict_events.csv 不存在")
        return

    df = pd.read_csv(conf_path)
    counts = df["level"].value_counts()

    level_map = {"critical": "紧急", "warning": "预警", "caution": "注意"}
    level_colors = {"critical": "#e74c3c", "warning": "#f39c12", "caution": "#f1c40f"}
    labels = [level_map.get(k, k) for k in counts.index]
    colors = [level_colors.get(k, "#7f8c8d") for k in counts.index]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 左：柱状图
    bars = axes[0].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.5, width=0.5)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                     str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[0].set_title("冲突预警等级分布", fontsize=13)
    axes[0].set_ylabel("事件数量")
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(0, counts.max() * 1.2)

    # 右：时域分布
    if "time_sec" in df.columns:
        bins = np.linspace(df["time_sec"].min(), df["time_sec"].max(), 30)
        for lvl, color in level_colors.items():
            sub = df[df["level"] == lvl]["time_sec"]
            if not sub.empty:
                axes[1].hist(sub.values, bins=bins, alpha=0.6, color=color,
                             label=level_map.get(lvl, lvl), edgecolor="none")
        axes[1].set_xlabel("时间 (s)")
        axes[1].set_ylabel("事件数")
        axes[1].set_title("冲突预警时域分布", fontsize=13)
        axes[1].legend(fontsize=11)
        axes[1].spines[["top", "right"]].set_visible(False)

    plt.suptitle("图4  冲突检测结果分析", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(fig_dir, "fig4_conflict.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  ✓ {path}")


def fig5_temporal_flow(output_dir, fig_dir):
    """图5: 时域流量曲线"""
    flow_path = os.path.join(output_dir, "temporal_flow.csv")
    if not os.path.exists(flow_path):
        print("  跳过 fig5：temporal_flow.csv 不存在")
        return

    df = pd.read_csv(flow_path)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()

    ax1.fill_between(df["time_sec"], df["active_targets"], alpha=0.25,
                     color="#3498db", label="活跃目标数")
    ax1.plot(df["time_sec"], df["active_targets"], color="#3498db", linewidth=1.5)
    ax1.set_xlabel("时间 (s)", fontsize=12)
    ax1.set_ylabel("活跃目标数", color="#3498db", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="#3498db")

    ax2.plot(df["time_sec"], df["total_reports"], color="#e74c3c",
             linewidth=1.2, linestyle="--", label="总报告数", alpha=0.7)
    ax2.set_ylabel("总报告数", color="#e74c3c", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#e74c3c")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=11)

    ax1.set_title("图5  低空目标时域流量分析", fontsize=14, pad=10)
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)

    # 标注峰值
    peak_idx = df["active_targets"].idxmax()
    peak_t = df.loc[peak_idx, "time_sec"]
    peak_v = df.loc[peak_idx, "active_targets"]
    ax1.annotate(f"峰值 {peak_v} 个", xy=(peak_t, peak_v),
                 xytext=(peak_t + 100, peak_v * 1.05),
                 arrowprops=dict(arrowstyle="->", color="gray"),
                 fontsize=10, color="#2c3e50")

    plt.tight_layout()
    path = os.path.join(fig_dir, "fig5_temporal_flow.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  ✓ {path}")


def fig6_fusion_pipeline(fig_dir):
    """图6: 系统架构流程图（示意）"""
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")

    steps = [
        ("ADS-B\nRemote ID\n雷达\n频谱", "#2c3e50", "white"),
        ("时空\n对齐", "#3498db", "white"),
        ("多源\n关联", "#9b59b6", "white"),
        ("加权\n融合", "#f39c12", "white"),
        ("目标\n分类", "#2ecc71", "white"),
        ("异常\n检测", "#e74c3c", "white"),
        ("态势\n感知", "#1a253a", "white"),
    ]

    box_w, box_h = 1.2, 1.4
    gap = 1.5
    y_center = 1.5

    for i, (label, color, fgcolor) in enumerate(steps):
        x = 0.5 + i * gap
        rect = mpatches.FancyBboxPatch(
            (x - box_w / 2, y_center - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.05", linewidth=1,
            edgecolor="white", facecolor=color
        )
        ax.add_patch(rect)
        ax.text(x, y_center, label, ha="center", va="center",
                fontsize=9, color=fgcolor, fontweight="bold")
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + box_w / 2 + 0.25, y_center),
                        xytext=(x + box_w / 2, y_center),
                        arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.5))

    ax.set_title("图6  低空目标多元融合监视系统流水线", fontsize=14, pad=10)
    plt.tight_layout()
    path = os.path.join(fig_dir, "fig6_pipeline.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  ✓ {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./output")
    parser.add_argument("--fig", default="./output/figures")
    args = parser.parse_args()

    os.makedirs(args.fig, exist_ok=True)
    print(f"生成论文配图到: {args.fig}")

    fig1_source_stats(args.output, args.fig)
    fig2_classification(args.output, args.fig)
    fig3_anomaly_distribution(args.output, args.fig)
    fig4_conflict_levels(args.output, args.fig)
    fig5_temporal_flow(args.output, args.fig)
    fig6_fusion_pipeline(args.fig)

    print(f"\n全部配图已生成到 {args.fig}/")


if __name__ == "__main__":
    main()
