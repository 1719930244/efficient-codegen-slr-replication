#!/usr/bin/env python3
"""fig-pareto.pdf v2 — A800 5轮数据, 19配置(含32B与投机解码格), p50延迟口径。

改动 vs v1 脚本:
- run1-5 聚合(P16/P17在run1-4, P18/P19在run6-7, 均为a800-1合并树)
- latency 用 p50_total_time_ms(与正文表格口径一致)
- 投机解码格用三角形标记
- 标签交替上下偏移防重叠(回应审稿人2 Presentation意见)
"""
import json, statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path.home() / "efficient-codegen-exp" / "results"
OUT = Path.home() / "efficient-codegen-exp" / "figures" / "fig-pareto-a800.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def mean(xs):
    xs = [x for x in xs if x is not None and float(x) >= 0]
    return statistics.mean(xs) if xs else None


RUNS_MAIN = ["run1", "run2", "run3", "run4", "run5"]
RUNS_32B = ["run1", "run2", "run3", "run4"]
RUNS_SPEC = ["run6", "run7"]

ids_std = [f"P{i:02d}" for i in range(1, 16)]          # run1-5
ids_32b = ["P16", "P17"]                                 # run1-4
ids_spec = ["P18", "P19"]                                # run6-7

rows = []
for pid, runs in [(p, RUNS_MAIN) for p in ids_std] + \
                 [(p, RUNS_32B) for p in ids_32b] + \
                 [(p, RUNS_SPEC) for p in ids_spec]:
    rs = [load(BASE / "pareto" / r / f"{pid}.json") for r in runs]
    rs = [r for r in rs if r]
    if not rs:
        print(f"skip {pid}: no run files")
        continue
    eround = load(BASE / "energy_round" / "pareto" / f"{pid}.json")
    energy = eround["mean_energy_j_per_request"] if eround and eround.get("mean_energy_j_per_request", -1) >= 0 else None
    if energy is None:
        energy = mean([r.get("mean_energy_j_per_request") for r in rs])
    rows.append(dict(
        pid=pid,
        model=rs[0]["model_name"],
        prec=rs[0]["precision"],
        decoding=rs[0].get("decoding", "standard"),
        passat1=mean([r["pass_at_1"] for r in rs]),
        latency=mean([r["p50_total_time_ms"] for r in rs]),
        tps=mean([r["mean_tokens_per_sec"] for r in rs]),
        memory=mean([r["peak_gpu_memory_mb"] for r in rs]),
        energy=energy,
        nruns=len(rs),
    ))

print(f"loaded {len(rows)} configs")
for r in rows:
    print(f"  {r['pid']} {r['model']:32s} {r['prec']:5s} {r['decoding']:12s} n={r['nruns']} pass={r['passat1']:.2f} p50={r['latency']:.0f} E={r['energy']}")

size_order = ["0.5B", "1.5B", "3B", "7B", "14B", "32B"]
def scale_of(name):
    for s in size_order:
        if s in name:
            return s
    return "?"

prec_color = {"fp16": "#2b6cb0", "int8": "#dd6b20", "int4": "#c53030"}
prec_label = {"fp16": "FP16", "int8": "INT8", "int4": "INT4"}


def pareto_front(points, lower_x_better=True):
    pts = sorted(points, key=lambda p: (p[0] if lower_x_better else -p[0]))
    front, best_y = [], -1
    for x, y in pts:
        if y > best_y:
            front.append((x, y))
            best_y = y
    return front


fig, axes = plt.subplots(2, 2, figsize=(11, 9))
configs = [
    (axes[0, 0], "latency", "Median Latency per Request (ms)", True),
    (axes[0, 1], "memory",  "Peak GPU Memory (MB)", True),
    (axes[1, 0], "tps",     "Throughput (tokens/s)", False),
    (axes[1, 1], "energy",  "Energy per Request (J)", True),
]

for ax, key, xlabel, lower_better in configs:
    pts = []
    li = 0
    for r in rows:
        if r[key] is None:
            continue
        x, y = r[key], r["passat1"]
        pts.append((x, y, r))
        spec = r["decoding"] == "speculative"
        ax.scatter(x, y,
                   marker="^" if spec else "o",
                   color=prec_color[r["prec"]],
                   s=100 if not spec else 110, zorder=5,
                   edgecolors="black", linewidth=0.6)
        label = f"{scale_of(r['model'])}-{prec_label[r['prec']]}" + ("-spec" if spec else "")
        dy = 6 if li % 2 == 0 else -12   # 交替上下防重叠
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(5, dy), fontsize=7.5, color="#333")
        li += 1
        if scale_of(r["model"]) == "7B" and r["prec"] == "int4" and not spec:
            ax.scatter(x, y, s=280, facecolors="none",
                       edgecolors="#6b46c1", linewidths=2.2, zorder=4)
    front = pareto_front([(p[0], p[1]) for p in pts], lower_x_better=lower_better)
    if front:
        fx, fy = zip(*front)
        ax.plot(fx, fy, "--", color="#555", alpha=0.7, linewidth=1.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("pass@1 (%)")
    ax.grid(True, alpha=0.3)

import matplotlib.patches as mpatches
import matplotlib.lines as mlines
legend_elems = [
    mpatches.Patch(color=prec_color["fp16"], label="FP16"),
    mpatches.Patch(color=prec_color["int8"], label="INT8"),
    mpatches.Patch(color=prec_color["int4"], label="INT4"),
    mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                  markeredgecolor="black", markersize=9, linestyle="", label="Standard decoding"),
    mlines.Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
                  markeredgecolor="black", markersize=9, linestyle="", label="Speculative decoding"),
    mlines.Line2D([0], [0], linestyle="--", color="#555", label="Pareto front"),
    mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                  markeredgecolor="#6b46c1", markersize=12, markeredgewidth=2,
                  linestyle="", label="7B-INT4 tracker"),
]
fig.legend(handles=legend_elems, loc="upper center", ncol=7, frameon=False,
           bbox_to_anchor=(0.5, 1.02), fontsize=9.5)

fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, bbox_inches="tight")
print(f"Saved {OUT}")
