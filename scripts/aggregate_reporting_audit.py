"""Regenerate the reporting-audit rates of the manuscript.

Reads data/reporting-audit/reporting-audit-117.csv and prints the overall
rates (Table 5), the reporting-breadth counts, the co-reporting counts, and
the per-stage rates (Table 8), all over the 117 studies of the final corpus.

Stages follow the canonical category mapping used for the RQ marginals:
1* and 2c and 3l map to RQ1, 2a/2b/2d/2f to RQ2, 3a/3b/3c/3d/3e/3f/3j/3k/3n/3o
to RQ3, 3g/3h/3i/3m to RQ4, and 4* to RQ5.
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "data" / "reporting-audit" / "reporting-audit-117.csv"
CORPUS = ROOT / "data" / "primary-studies-117.csv"

ITEMS = ["Correctness", "ModelName", "Hardware", "Latency", "Memory",
         "ServingConfig", "MonetaryCost", "Energy"]
DEPLOY = {"3g", "3h", "3i", "3m"}
DATA = {"2c", "3l"}


def code_to_rq(code):
    code = code.strip().split("-")[0].strip()
    if code[:2] in DATA:
        return "RQ1"
    if not code or not code[0].isdigit():
        return None
    return {"1": "RQ1", "2": "RQ2", "4": "RQ5",
            "3": "RQ4" if code[:2] in DEPLOY else "RQ3"}[code[0]]


def main():
    audit = {r["ID"]: r for r in csv.DictReader(open(AUDIT, encoding="utf-8-sig"))}
    corpus = list(csv.DictReader(open(CORPUS, encoding="utf-8-sig")))
    n = len(corpus)

    print(f"Over {n} studies")
    for item in ITEMS:
        yes = sum(1 for r in corpus if audit[r["ID"]][item] == "YES")
        print(f"  {item:14s} {yes:3d}/{n}  {100 * yes / n:5.1f}%")

    breadth = Counter(int(audit[r["ID"]]["NEfficiencyMetrics"] or 0) for r in corpus)
    print("\nBreadth:", {k: breadth[k] for k in sorted(breadth)})
    both = Counter(audit[r["ID"]]["BothLatencyMemory"] for r in corpus)
    print("Latency and memory together:", both["YES"])

    stages = defaultdict(list)
    for r in corpus:
        codes = [c.strip() for c in r["Primary Categories"].replace(";", ",").split(",") if c.strip()]
        for stage in set(filter(None, (code_to_rq(c) for c in codes))):
            stages[stage].append(audit[r["ID"]])

    print("\nBy stage")
    for stage in ["RQ1", "RQ2", "RQ3", "RQ4", "RQ5"]:
        rows = stages[stage]
        k = len(rows)
        rates = " ".join(
            f"{item[:4]} {100 * sum(1 for r in rows if r[item] == 'YES') / k:4.0f}%"
            for item in ["Latency", "Memory", "Energy", "MonetaryCost"])
        print(f"  {stage}  N={k:3d}  {rates}")


if __name__ == "__main__":
    main()