#!/usr/bin/env python3
"""fig-pareto.pdf v2 — A800 5轮数据, 19配置(含32B与投机解码格), p50延迟口径。

改动 vs v1 脚本:
- run1-5 聚合(P16/P17在run1-4, P18/P19在run6-7, 均为a800-1合并树)
- latency 用 p50_total_time_ms(与正文表格口径一致)
- 投机解码格用三角形标记
- 标签贪心排布防重叠，偏离数据点的标签用引线连回
"""
import os
import json, statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(os.environ.get("SLR_A800_DATA", str(Path(__file__).resolve().parents[1] / "data" / "a800-results" / "a800-1")))
OUT = Path(os.environ.get("SLR_PARETO_OUT", str(Path(__file__).resolve().parents[1] / "figures" / "fig-pareto-a800.pdf")))
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
        memory=min(r["peak_gpu_memory_mb"] for r in rs if r.get("peak_gpu_memory_mb",-1)>=0),
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

# Candidate label offsets (points). Labels are placed greedily, most crowded point first,
# at the candidate with the least overlap; a leader line joins any label set away from its point.
CANDIDATES = [(5, 5), (5, -5), (-5, 5), (-5, -5), (0, 10), (0, -10),
              (14, 12), (14, -14), (-14, 12), (-14, -14), (22, 0), (-22, 0),
              (18, 24), (18, -26), (-18, 24), (-18, -26), (34, 10), (-34, 10),
              (34, -18), (-34, -18), (8, 36), (8, -38), (-8, 36), (-8, -38),
              (46, 22), (-46, 22), (46, -30), (-46, -30),
              (62, 0), (62, -14), (62, 14), (62, -28), (78, -8), (78, 8), (78, -22)]
LEADER_MIN_PT = 9
FONT = 7.5


def _ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def crosses(s1, s2):
    a, b = s1; c, d = s2
    return _ccw(a, c, d) != _ccw(b, c, d) and _ccw(a, b, c) != _ccw(a, b, d)


def overlap(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def place_labels(ax, items):
    """items: list of (x, y, text). Returns list of (x, y, text, dx, dy, ha, va)."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    s = fig.dpi / 72.0
    ab = ax.get_window_extent(renderer)
    pts = [(ax.transData.transform((x, y)), t) for x, y, t in items]
    sizes = {}
    for _, t in pts:
        tt = ax.text(0, 0, t, fontsize=FONT)
        e = tt.get_window_extent(renderer)
        sizes[t] = (e.width, e.height)
        tt.remove()
    marker_r = 7.5 * s
    markers = [(p[0] - marker_r, p[1] - marker_r, p[0] + marker_r, p[1] + marker_r) for p, _ in pts]
    # crowded points first
    def crowd(i):
        p = pts[i][0]
        return -sum(1 for q, _ in pts if abs(q[0] - p[0]) < 60 * s and abs(q[1] - p[1]) < 30 * s)
    order = sorted(range(len(pts)), key=crowd)
    placed, segs, out = [], [], [None] * len(pts)
    for i in order:
        (px, py), t = pts[i]
        w, h = sizes[t]
        best = None
        for k, (dx, dy) in enumerate(CANDIDATES):
            ha = "left" if dx >= 0 else "right"
            va = "bottom" if dy >= 0 else "top"
            x0 = px + dx * s - (w if ha == "right" else 0)
            y0 = py + dy * s - (h if va == "top" else 0)
            if dx == 0:
                x0 = px - w / 2
                ha = "center"
            box = (x0, y0, x0 + w, y0 + h)
            pad = 2.5 * s
            pbox = (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)
            ov = sum(overlap(pbox, b) for b in placed)
            cost = ov * 4 + (20000 * s * s if ov > 0 else 0)
            cost += sum(overlap(box, m) for j, m in enumerate(markers) if j != i) * 2
            for sg in segs:
                (sx, sy), (ex, ey) = sg
                if any(box[0] <= sx + (ex - sx) * u / 12 <= box[2] and box[1] <= sy + (ey - sy) * u / 12 <= box[3] for u in range(1, 12)):
                    cost += 3000 * s * s
            out_x = max(0, ab.x0 + 2 - box[0]) + max(0, box[2] - ab.x1 + 2)
            out_y = max(0, ab.y0 + 2 - box[1]) + max(0, box[3] - ab.y1 + 2)
            cost += (out_x * h + out_y * w) * 6 + (20000 * s * s if out_x or out_y else 0)
            cost += k * 0.5 * s * s          # prefer near, earlier candidates
            # a label without a leader line must sit closer to its own point than to any other
            if (dx * dx + dy * dy) ** 0.5 < LEADER_MIN_PT:
                def dist(q):
                    cx = min(max(q[0], box[0]), box[2]); cy = min(max(q[1], box[1]), box[3])
                    return ((cx - q[0]) ** 2 + (cy - q[1]) ** 2) ** 0.5
                own = dist((px, py))
                if any(dist(q) < own + 3 * s for j, (q, _) in enumerate(pts) if j != i):
                    cost += 5000 * s * s
            seg = None
            if (dx * dx + dy * dy) ** 0.5 >= LEADER_MIN_PT:
                ax_ = min(max(px, box[0]), box[2]); ay_ = min(max(py, box[1]), box[3])
                seg = ((px, py), (ax_, ay_))
                cost += sum(1 for sg in segs if crosses(seg, sg)) * 3000 * s * s
                samples = [(px + (ax_ - px) * u / 12, py + (ay_ - py) * u / 12) for u in range(1, 12)]
                for bx in placed:   # leader through another label
                    if any(bx[0] <= qx <= bx[2] and bx[1] <= qy <= bx[3] for qx, qy in samples):
                        cost += 3000 * s * s
                for j, (q, _) in enumerate(pts):   # leader over another dot
                    if j != i and any((qx - q[0]) ** 2 + (qy - q[1]) ** 2 < (marker_r * 0.9) ** 2 for qx, qy in samples):
                        cost += 3000 * s * s
            if best is None or cost < best[0]:
                best = (cost, dx, dy, ha, va, box, seg)
        _, dx, dy, ha, va, box, seg = best
        placed.append(box)
        if seg:
            segs.append(seg)
        x, y, _ = items[i]
        out[i] = (x, y, t, dx, dy, ha, va)
    return out


pending = []
for ax, key, xlabel, lower_better in configs:
    pts = []
    labels = []
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
        labels.append((x, y, f"{scale_of(r['model'])}-{prec_label[r['prec']]}" + ("-spec" if spec else "")))
        if scale_of(r["model"]) == "7B" and r["prec"] == "int4" and not spec:
            ax.scatter(x, y, s=280, facecolors="none",
                       edgecolors="#6b46c1", linewidths=2.2, zorder=4)
    front = pareto_front([(p[0], p[1]) for p in pts], lower_x_better=lower_better)
    if front:
        fx, fy = zip(*front)
        ax.plot(fx, fy, "--", color="#555", alpha=0.7, linewidth=1.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Pass@1 (%)")
    ax.grid(True, alpha=0.3)
    ax.margins(x=0.08, y=0.08)
    pending.append((ax, labels))


def draw_labels():
    for ax, labels in pending:
      for x, y, t, dx, dy, ha, va in place_labels(ax, labels):
          far = (dx * dx + dy * dy) ** 0.5 >= LEADER_MIN_PT
          ax.annotate(t, (x, y), textcoords="offset points", xytext=(dx, dy),
                      ha=ha, va=va, fontsize=FONT, color="#333", zorder=6,
                      arrowprops=dict(arrowstyle="-", color="#666", lw=0.7, shrinkA=0, shrinkB=5) if far else None)


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
draw_labels()
fig.savefig(OUT, bbox_inches="tight")
print(f"Saved {OUT}")
