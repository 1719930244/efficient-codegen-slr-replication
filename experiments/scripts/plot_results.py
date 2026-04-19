"""可视化脚本：生成 Pareto 图和 Composition 对比图

用法:
    uv run python scripts/plot_results.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.size"] = 12
matplotlib.rcParams["figure.dpi"] = 150

RESULT_DIR = Path.home() / "efficient-codegen-exp" / "results"
FIGURE_DIR = Path.home() / "efficient-codegen-exp" / "figures"


def load_results(subdir: str, runs: int = 3) -> list[dict]:
    """加载多轮结果并取平均"""
    all_configs = {}

    for run_idx in range(1, runs + 1):
        run_dir = RESULT_DIR / subdir / f"run{run_idx}"
        summary = run_dir / "summary.json"
        if not summary.exists():
            continue
        with open(summary) as f:
            data = json.load(f)
        for item in data:
            cid = item["config_id"]
            if cid not in all_configs:
                all_configs[cid] = []
            all_configs[cid].append(item)

    # 取平均
    averaged = []
    for cid, runs_data in all_configs.items():
        avg = runs_data[0].copy()
        numeric_keys = [
            "pass_at_1", "mean_ms_per_token", "mean_total_time_ms",
            "p50_total_time_ms", "p95_total_time_ms",
            "mean_tokens_per_sec", "peak_gpu_memory_mb",
        ]
        for key in numeric_keys:
            values = [r[key] for r in runs_data if key in r]
            avg[key] = np.mean(values) if values else 0
        avg["details"] = []  # 不保留详细数据
        averaged.append(avg)

    return averaged


# ── Pareto 图 ──────────────────────────────────────────────

def plot_pareto():
    """绘制 Pareto 前沿图（3 张）"""
    results = load_results("pareto")
    if not results:
        print("No Pareto results found.")
        return

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # 颜色和标记：按模型尺寸
    size_order = ["0.5B", "1.5B", "3B", "7B", "14B"]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(size_order)))
    size_color = {s: c for s, c in zip(size_order, colors)}
    prec_marker = {"fp16": "o", "int8": "s", "int4": "^"}

    metrics = [
        ("mean_total_time_ms", "Mean Latency (ms)", "pareto_latency.pdf"),
        ("peak_gpu_memory_mb", "Peak GPU Memory (MB)", "pareto_memory.pdf"),
        ("mean_tokens_per_sec", "Throughput (tokens/s)", "pareto_throughput.pdf"),
    ]

    for metric_key, xlabel, filename in metrics:
        fig, ax = plt.subplots(figsize=(8, 6))

        for r in results:
            model_name = r["model_name"]
            # 从模型名中提取尺寸
            size = None
            for s in size_order:
                if s in model_name:
                    size = s
                    break
            if size is None:
                continue

            x = r[metric_key]
            y = r["pass_at_1"]
            prec = r["precision"]

            ax.scatter(x, y,
                       color=size_color[size], marker=prec_marker.get(prec, "o"),
                       s=100, zorder=5, edgecolors="black", linewidth=0.5)
            ax.annotate(f"{size}-{prec}", (x, y),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

        # Pareto 前沿线
        points = [(r[metric_key], r["pass_at_1"]) for r in results]
        if "throughput" in filename:
            # 吞吐量越高越好
            pareto = compute_pareto_front(points, higher_x_better=True)
        else:
            # 延迟/内存越低越好
            pareto = compute_pareto_front(points, higher_x_better=False)
        if pareto:
            px, py = zip(*pareto)
            ax.plot(px, py, "r--", alpha=0.5, linewidth=1.5, label="Pareto front")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("pass@1 (%)")
        ax.set_title(f"Efficiency-Quality Trade-off: {xlabel.split('(')[0].strip()}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(FIGURE_DIR / filename, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {filename}")


def compute_pareto_front(points, higher_x_better=False):
    """计算 2D Pareto 前沿点（pass@1 越高越好）"""
    if not points:
        return []
    # 按 x 排序
    if higher_x_better:
        sorted_pts = sorted(points, key=lambda p: -p[0])  # x 降序（越高越好）
    else:
        sorted_pts = sorted(points, key=lambda p: p[0])   # x 升序（越低越好）
    # 扫描：保留 pass@1 单调不减的点
    front = [sorted_pts[0]]
    for x, y in sorted_pts[1:]:
        if y >= front[-1][1]:
            front.append((x, y))
    return sorted(front, key=lambda p: p[0])


# ── Composition 对比图 ──────────────────────────────────

def plot_composition():
    """绘制技术组合对比图"""
    results = load_results("composition")
    if not results:
        print("No Composition results found.")
        return

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # 按 config_id 排序
    results.sort(key=lambda r: r["config_id"])

    config_ids = [r["config_id"] for r in results]
    pass1 = [r["pass_at_1"] for r in results]
    latency = [r["mean_total_time_ms"] for r in results]
    memory = [r["peak_gpu_memory_mb"] for r in results]
    throughput = [r["mean_tokens_per_sec"] for r in results]

    # 图 1：pass@1 + latency 双轴
    fig, ax1 = plt.subplots(figsize=(12, 5))
    x = np.arange(len(config_ids))
    width = 0.35

    bars1 = ax1.bar(x - width/2, pass1, width, label="pass@1 (%)", color="#2196F3")
    ax1.set_ylabel("pass@1 (%)", color="#2196F3")
    ax1.set_ylim(0, max(pass1) * 1.3 if pass1 else 100)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, latency, width, label="Latency (ms)", color="#FF9800", alpha=0.7)
    ax2.set_ylabel("Mean Latency (ms)", color="#FF9800")

    ax1.set_xticks(x)
    ax1.set_xticklabels(config_ids, rotation=45)
    ax1.set_xlabel("Configuration")
    ax1.set_title("Technique Composition: Quality vs Latency")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "composition_quality_latency.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved composition_quality_latency.pdf")

    # 图 2：内存对比
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = []
    for r in results:
        if r["precision"] == "fp16":
            colors.append("#4CAF50")
        elif r["precision"] == "int8":
            colors.append("#2196F3")
        else:
            colors.append("#FF5722")
    ax.bar(config_ids, memory, color=colors)
    ax.set_ylabel("Peak GPU Memory (MB)")
    ax.set_xlabel("Configuration")
    ax.set_title("Technique Composition: Memory Usage")
    ax.tick_params(axis="x", rotation=45)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", label="FP16"),
        Patch(facecolor="#2196F3", label="INT8"),
        Patch(facecolor="#FF5722", label="INT4"),
    ]
    ax.legend(handles=legend_elements)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "composition_memory.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved composition_memory.pdf")

    # 图 3：加性分析热力图
    plot_interaction_heatmap(results)


def plot_interaction_heatmap(results):
    """绘制技术交互效应热力图"""
    # 构建 speedup 矩阵：以 C01 (FP16+Standard+Greedy) 为基准
    baseline = next((r for r in results if r["config_id"] == "C01"), None)
    if not baseline:
        return

    base_latency = baseline["mean_total_time_ms"]
    if base_latency == 0:
        return

    # 按维度分组计算 speedup
    precisions = ["fp16", "int8", "int4"]
    decodings = ["standard", "speculative"]

    fig, ax = plt.subplots(figsize=(6, 4))

    matrix = np.zeros((len(precisions), len(decodings)))
    for r in results:
        if r["sampling"] != "greedy":
            continue
        pi = precisions.index(r["precision"]) if r["precision"] in precisions else -1
        di = decodings.index(r["decoding"]) if r["decoding"] in decodings else -1
        if pi >= 0 and di >= 0:
            matrix[pi][di] = base_latency / r["mean_total_time_ms"] if r["mean_total_time_ms"] > 0 else 0

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(decodings)))
    ax.set_xticklabels(["Standard", "Speculative"])
    ax.set_yticks(range(len(precisions)))
    ax.set_yticklabels(["FP16", "INT8", "INT4"])
    ax.set_xlabel("Decoding Strategy")
    ax.set_ylabel("Precision")
    ax.set_title("Speedup over Baseline (Greedy)")

    for i in range(len(precisions)):
        for j in range(len(decodings)):
            ax.text(j, i, f"{matrix[i][j]:.2f}x",
                    ha="center", va="center", fontsize=14, fontweight="bold")

    fig.colorbar(im, label="Speedup")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "composition_interaction_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved composition_interaction_heatmap.pdf")


def main():
    print("=== Generating Figures ===\n")
    plot_pareto()
    print()
    plot_composition()
    print("\nAll figures saved to:", FIGURE_DIR)


if __name__ == "__main__":
    main()
