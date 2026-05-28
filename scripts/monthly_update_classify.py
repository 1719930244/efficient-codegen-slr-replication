"""Dedupe against existing primary studies and bucket candidates by RQ."""

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
PRIMARY_CSV = ROOT / "data" / "primary-studies.csv"
CANDIDATES = ROOT / "data" / "monthly-update-2026-05" / "s2_filtered.json"
OUT_DIR = ROOT / "data" / "monthly-update-2026-05"


def load_existing_keys():
    keys = set()
    titles = set()
    with PRIMARY_CSV.open() as f:
        for row in csv.DictReader(f):
            keys.add(row["Key"].strip())
            titles.add(row["Title"].strip().lower())
    # canonicalize arXiv keys: arxiv_2407_05040 -> 2407.05040, 2407_05040 -> 2407.05040
    canon = set()
    for k in keys:
        k = k.lstrip("arxiv_")
        k = k.replace("_", ".")
        canon.add(k)
    return canon, titles


RQ_BUCKETS = [
    ("RQ1 数据", [
        r"\bdata.{0,15}(efficient|pruning|selection|synthesis|curat|quality|distill|augment)",
        r"\b(curriculum|instruction).{0,5}tuning",
        r"\bdata.{0,5}pruning",
    ]),
    ("RQ2 训练", [
        r"\b(fine-?tun|distill|knowledge.{0,3}distill|PEFT|LoRA|adapter|low-?rank)",
        r"\b(curriculum|reinforcement|RL).{0,3}(learn|tun|train)",
        r"\b(quantiz|pruning|model.{0,3}pruning).{0,10}(train|fine-?tun)",
    ]),
    ("RQ3 推理", [
        r"\b(speculative.{0,3}decod|spec.{0,3}decod|early.{0,3}exit|non-?autoregressive|NAR)",
        r"\b(KV.{0,3}cache|prompt.{0,3}compress|context.{0,3}prun|cache)",
        r"\b(quantiz|pruning|distill).{0,15}(inference|decod|serving)",
        r"\b(routing|model.{0,3}routing|adaptive.{0,3}sampl)",
        r"\b(multi-?agent|agent.{0,5}collaborat|chain-?of-?thought.{0,3}opt|CoT.{0,3}opt)",
        r"\b(code.{0,3}specific|token.{0,3}efficient|prompt.{0,3}eng)",
        r"\b(latency|throughput|inference.{0,5}efficien|hide.{0,5}latency|accelerat)",
    ]),
    ("RQ4 部署", [
        r"\b(serving|deploy|edge|on-?device|mobile|cloud|GPU|hardware)",
        r"\bSLO|SLA|batch|prefill|frequency.{0,3}scal",
    ]),
    ("RQ5 评估", [
        r"\bbenchmark|empirical.{0,3}study|evaluation|metric|pareto|frontier|token.{0,3}consump|cost",
    ]),
]
RQ_REGEX = [(name, [re.compile(p, re.IGNORECASE) for p in pats]) for name, pats in RQ_BUCKETS]


def classify(text):
    hits = []
    for name, regs in RQ_REGEX:
        if any(r.search(text) for r in regs):
            hits.append(name)
    return hits or ["未分类"]


TITLE_EFFICIENCY_RE = re.compile(
    r"\b(efficien|optimi|accelerat|lightweight|compress|quantiz|prun|distill|"
    r"latency|throughput|computational.{0,3}cost|energy|scalab|speedup|"
    r"cache|MoE|sparse|low-?rank|LoRA|adapter|PEFT|spec|early.{0,3}exit|"
    r"token.{0,3}efficien|cost-?effic|fast|memory)",
    re.IGNORECASE,
)


def main():
    existing_keys, existing_titles = load_existing_keys()
    candidates = json.loads(CANDIDATES.read_text())
    print(f"Loaded {len(candidates)} boolean-filter candidates")
    print(f"Existing corpus: {len(existing_keys)} keys, {len(existing_titles)} titles")

    fresh = []
    duplicates = []
    for p in candidates:
        ext = p.get("externalIds") or {}
        arx = ext.get("ArXiv") or ""
        title = (p.get("title") or "").strip().lower()
        if arx and arx.replace("_", ".") in existing_keys:
            duplicates.append(p)
            continue
        if title in existing_titles:
            duplicates.append(p)
            continue
        # Stricter: title must signal efficiency, not just abstract.
        if not TITLE_EFFICIENCY_RE.search(p.get("title") or ""):
            continue
        fresh.append(p)

    print(f"Fresh + title-has-efficiency-signal: {len(fresh)}")
    print(f"Duplicates with existing: {len(duplicates)}")

    # Bucket by RQ
    bucketed = {name: [] for name, _ in RQ_BUCKETS}
    bucketed["未分类"] = []
    for p in fresh:
        title = p.get("title") or ""
        abstract = p.get("abstract") or ""
        text = title + " " + abstract
        for tag in classify(text):
            bucketed.setdefault(tag, []).append(p)

    (OUT_DIR / "fresh_candidates.json").write_text(json.dumps(fresh, ensure_ascii=False, indent=2))

    # Summary table
    print("\n--- Fresh candidates by RQ (Boolean filter passed) ---")
    for name in [n for n, _ in RQ_BUCKETS] + ["未分类"]:
        papers = bucketed.get(name, [])
        if not papers:
            continue
        print(f"\n{name} ({len(papers)} papers):")
        for p in sorted(papers, key=lambda x: x.get("publicationDate") or ""):
            arx = (p.get("externalIds") or {}).get("ArXiv", "-")
            date = (p.get("publicationDate") or "?")[:10]
            title = (p.get("title") or "")[:95]
            print(f"  {date}  arxiv={arx:14s}  {title}")


if __name__ == "__main__":
    main()
