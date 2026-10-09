"""UpSet plot of primary studies across the five lifecycle RQs.

Reads data/primary-studies.csv of the replication package (the single
source of truth for the merged 141-study corpus; override with UPSET_CSV,
the default path below is the sg hub clone) and maps the Primary
Categories codes to the five lifecycle stages of the per-stage compliance
table (Table tab:compliance-stage):

  1*, 2c, 3l                        -> RQ1 Data
  2a/2b/2d/2f                       -> RQ2 Training
  3a/3b/3c/3d/3e/3f/3j/3k/3n/3o     -> RQ3 Inference
  3g/3h/3i/3m                       -> RQ4 Deployment
  4*                                -> RQ5 Evaluation

This rule reproduces the stage head counts 29/26/63/28/27 of that table
and its 32 multi-stage studies exactly (122-study vintage: 25/24/59/24/21
and 31). This mapping, not the CSV's legacy RQ column, is canonical for
the stage marginals; 2c (distillation) and 3l (code-specific
optimization) attach to the data stage here and are discussed in the
chapters of the stages their methods target.

RQ6 is the controlled-experiment question and does not classify any
primary study, so it is excluded from this plot. Output: figures/fig-upset.pdf.
"""

import csv
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CSV = Path(os.environ.get("UPSET_CSV", str(ROOT / "data" / "primary-studies.csv")))
OUT = ROOT / "figures" / "fig-upset.pdf"

RQS = ["RQ1", "RQ2", "RQ3", "RQ4", "RQ5"]
RQ_LABELS = {
    "RQ1": "RQ1 Data",
    "RQ2": "RQ2 Training",
    "RQ3": "RQ3 Inference",
    "RQ4": "RQ4 Deployment",
    "RQ5": "RQ5 Evaluation",
}

DEPLOY_CODES = {"3g", "3h", "3i", "3m"}
DATA_CODES = {"2c", "3l"}


def code_to_rq(code):
    code = code.strip().split("-")[0].strip()
    if code[:2] in DATA_CODES:
        return "RQ1"
    if not code or not code[0].isdigit():
        return None
    digit = code[0]
    if digit == "1":
        return "RQ1"
    if digit == "2":
        return "RQ2"
    if digit == "3":
        prefix = code[:2]
        return "RQ4" if prefix in DEPLOY_CODES else "RQ3"
    if digit == "4":
        return "RQ5"
    return None


def load_assignments():
    combo_counter = Counter()
    rq_counter = Counter()
    n_studies = 0
    n_multi = 0
    with CSV.open() as f:
        for row in csv.DictReader(f):
            rqs = set()
            for field in ("Primary Categories",):
                for c in row[field].replace(",", ";").split(";"):
                    rq = code_to_rq(c)
                    if rq:
                        rqs.add(rq)
            if not rqs:
                continue
            n_studies += 1
            if len(rqs) > 1:
                n_multi += 1
            combo_counter[tuple(sorted(rqs))] += 1
            for rq in rqs:
                rq_counter[rq] += 1
    return combo_counter, rq_counter, n_studies, n_multi


def main():
    combos, rq_counts, n_studies, n_multi = load_assignments()
    ordered = combos.most_common()
    n_combos = len(ordered)

    fig = plt.figure(figsize=(10.0, 4.9))
    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        width_ratios=[1.5, 1.35, 7.0],
        height_ratios=[2.4, 1.9],
        wspace=0.16, hspace=0.05,
    )

    ax_bar = fig.add_subplot(gs[0, 2])
    ax_dots = fig.add_subplot(gs[1, 2])
    ax_labels = fig.add_subplot(gs[1, 1])
    ax_left = fig.add_subplot(gs[1, 0])

    xs = list(range(n_combos))
    sizes = [c for _, c in ordered]
    ax_bar.bar(xs, sizes, width=0.6, color="#3B7DD8", edgecolor="#1f4f8c")
    for x, s in zip(xs, sizes):
        ax_bar.text(x, s + max(sizes) * 0.02, str(s),
                    ha="center", va="bottom", fontsize=8)
    ax_bar.set_ylabel("Studies in intersection", fontsize=9)
    ax_bar.set_xlim(-0.7, n_combos - 0.3)
    ax_bar.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.set_ylim(0, max(sizes) * 1.18)

    rq_y = {rq: i for i, rq in enumerate(reversed(RQS))}
    for x, (combo, _) in enumerate(ordered):
        ys = sorted(rq_y[r] for r in combo)
        if len(ys) > 1:
            ax_dots.plot([x, x], [min(ys), max(ys)],
                         color="#1f4f8c", linewidth=1.5, zorder=1)
        for rq in RQS:
            y = rq_y[rq]
            color = "#1f4f8c" if rq in combo else "#d9d9d9"
            ax_dots.scatter([x], [y], s=60, color=color, zorder=2)
    ax_dots.set_xlim(-0.7, n_combos - 0.3)
    ax_dots.set_ylim(-0.6, len(RQS) - 0.4)
    ax_dots.set_xticks([])
    ax_dots.set_yticks([])
    for s in ("top", "right", "bottom", "left"):
        ax_dots.spines[s].set_visible(False)

    ax_labels.set_ylim(ax_dots.get_ylim())
    ax_labels.set_xlim(0, 1)
    for rq, y in rq_y.items():
        ax_labels.text(1.0, y, RQ_LABELS[rq],
                       ha="right", va="center", fontsize=9)
    ax_labels.axis("off")

    left_vals = [rq_counts[r] for r in reversed(RQS)]
    ax_left.barh(list(rq_y.values()), left_vals, height=0.55,
                 color="#E08A2C", edgecolor="#8a4f10")
    for y, v in zip(rq_y.values(), left_vals):
        ax_left.text(v + max(left_vals) * 0.03, y, str(v),
                     va="center", ha="right", fontsize=8)
    ax_left.set_ylim(ax_dots.get_ylim())
    ax_left.invert_xaxis()
    ax_left.set_xlabel("Set size", fontsize=9)
    ax_left.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["left"].set_visible(False)
    ax_left.set_xlim(max(left_vals) * 1.45, 0)

    fig.suptitle(
        f"Primary studies across RQ1--RQ5 "
        f"(N={n_studies}; {n_multi} span multiple RQs; RQ6 is experimental and not plotted)",
        fontsize=10, y=0.99,
    )

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT}")
    print(f"Studies: {n_studies}; multi-RQ: {n_multi}; intersections: {n_combos}")
    print("Per-RQ totals:", {rq: rq_counts[rq] for rq in RQS})
    for combo, n in ordered:
        print(f"  {' & '.join(combo)}: {n}")


if __name__ == "__main__":
    main()
