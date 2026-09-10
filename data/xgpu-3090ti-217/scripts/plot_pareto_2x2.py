#!/usr/bin/env python3
"""Generate fig-pareto.pdf as a 2x2 combined figure.

Subplots (row-major):
    (0,0) pass@1 vs Latency
    (0,1) pass@1 vs Memory
    (1,0) pass@1 vs Throughput
    (1,1) pass@1 vs Energy

Points labeled as "scale-precision"; precision colors per caption:
    FP16=blue, INT8=orange, INT4=red.
Dashed line connects Pareto-optimal points per subplot.
Purple ring marks 7B-INT4 across subplots.
"""

import json, statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path.home() / "efficient-codegen-exp" / "results"
OUT  = Path.home() / "efficient-codegen-exp" / "figures" / "fig-pareto.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

def load(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None

def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None

# Aggregate 3-run means for pareto main metrics
pareto_ids = [f"P{i:02d}" for i in range(1, 16)]
rows = []
for pid in pareto_ids:
    runs = [load(BASE/"pareto"/r/f"{pid}.json") for r in ("run1","run2","run3")]
    runs = [r for r in runs if r]
    if not runs: continue
    eround = load(BASE/"energy_round"/"pareto"/f"{pid}.json")
    rows.append(dict(
        model   = runs[0]["model_name"],
        prec    = runs[0]["precision"],
        passat1 = mean([r["pass_at_1"] for r in runs]),
        latency = mean([r["mean_total_time_ms"] for r in runs]),
        tps     = mean([r["mean_tokens_per_sec"] for r in runs]),
        memory  = mean([r["peak_gpu_memory_mb"] for r in runs]),
        energy  = eround["mean_energy_j_per_request"] if eround else None,
    ))

# Scale labels
size_order = ["0.5B", "1.5B", "3B", "7B", "14B"]
def scale_of(name):
    for s in size_order:
        if s in name: return s
    return "?"

# Precision colors (per caption)
prec_color = {"fp16": "#2b6cb0", "int8": "#dd6b20", "int4": "#c53030"}
prec_label = {"fp16": "FP16", "int8": "INT8", "int4": "INT4"}

def pareto_front(points, lower_x_better=True):
    """Return (x,y) pairs on Pareto front. y always higher-better (pass@1)."""
    pts = sorted(points, key=lambda p: (p[0] if lower_x_better else -p[0]))
    front, best_y = [], -1
    for x, y in pts:
        if y > best_y:
            front.append((x, y))
            best_y = y
    return front

fig, axes = plt.subplots(2, 2, figsize=(11, 9))

configs = [
    (axes[0,0], "latency", "Mean Latency (ms)",   True),
    (axes[0,1], "memory",  "Peak GPU Memory (MB)", True),
    (axes[1,0], "tps",     "Throughput (tokens/s)", False),
    (axes[1,1], "energy",  "Energy per Request (J)", True),
]

for ax, key, xlabel, lower_better in configs:
    pts = []
    for r in rows:
        if r[key] is None: continue
        x, y = r[key], r["passat1"]
        pts.append((x, y, r))
        ax.scatter(x, y,
                   color=prec_color[r["prec"]],
                   s=90, zorder=5,
                   edgecolors="black", linewidth=0.6)
        ax.annotate(f"{scale_of(r['model'])}-{prec_label[r['prec']]}",
                    (x, y),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=8, color="#333")
        # Purple ring on 7B-INT4
        if scale_of(r["model"]) == "7B" and r["prec"] == "int4":
            ax.scatter(x, y, s=260, facecolors="none",
                       edgecolors="#6b46c1", linewidths=2.2, zorder=4)
    # Pareto front
    front = pareto_front([(p[0], p[1]) for p in pts],
                         lower_x_better=lower_better)
    if front:
        fx, fy = zip(*front)
        ax.plot(fx, fy, "--", color="#555", alpha=0.7, linewidth=1.4,
                label="Pareto front")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("pass@1 (%)")
    ax.grid(True, alpha=0.3)

# Single legend in figure
import matplotlib.patches as mpatches
import matplotlib.lines  as mlines
legend_elems = [
    mpatches.Patch(color=prec_color["fp16"], label="FP16"),
    mpatches.Patch(color=prec_color["int8"], label="INT8"),
    mpatches.Patch(color=prec_color["int4"], label="INT4"),
    mlines.Line2D([0], [0], linestyle="--", color="#555", label="Pareto front"),
    mlines.Line2D([0], [0], marker="o", color="w",
                  markerfacecolor="none", markeredgecolor="#6b46c1",
                  markersize=12, markeredgewidth=2,
                  linestyle="", label="7B-INT4 tracker"),
]
fig.legend(handles=legend_elems, loc="upper center",
           ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02),
           fontsize=10)

fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, bbox_inches="tight")
print(f"Saved {OUT}")
