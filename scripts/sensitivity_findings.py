"""Peer-reviewed-subset sensitivity for Findings 1-7 (TOSEM R1, Section 3.8).

A study supports a finding when one of its Primary Categories is in the
finding's category set below. Peer-reviewed = Venue Type in {conference, journal}.
"""
import csv, os, sys
from pathlib import Path
CSV = os.environ.get("CSV", str(Path(__file__).resolve().parents[1] / "data" / "primary-studies.csv"))
SETS = {
    "F1": ["1a", "1b", "2d"],             # data selection, data quality, curriculum
    "F2": ["2a", "2b"],                   # efficient pre-training, PEFT
    "F3": ["2c", "2f"],                   # distillation, RL training
    "F4": ["3a", "3b", "3j", "3k", "3o"], # speculative, early exit, sampling, multi-agent gating, CoT
    "F5": ["3d", "3e", "3l"],             # prompt compression, context pruning, code-specific
    "F6": ["3g", "3h", "3i", "3m"],       # PTQ, pruning, routing, system optimization
    "F7": ["4a", "4b"],                   # benchmarks, empirical studies
}
rows = list(csv.DictReader(open(CSV)))
def cats(r):
    return {c.strip().split("-")[0] for c in r["Primary Categories"].split(";") if c.strip()}
print("finding,total,peer,preprint,peer_share")
for f, s in SETS.items():
    sup = [r for r in rows if cats(r) & set(s)]
    peer = sum(r["Venue Type"] in ("conference", "journal") for r in sup)
    print(f"{f},{len(sup)},{peer},{len(sup)-peer},{peer/len(sup):.2f}")
peer_all = [r for r in rows if r["Venue Type"] in ("conference", "journal")]
print("peer-reviewed total", len(peer_all))

# Per-RQ peer share under the canonical stage mapping of scripts/plot_upset.py
RQ = {"RQ1": ["1a", "1b", "2c", "3l"], "RQ2": ["2a", "2b", "2d", "2f"],
      "RQ3": ["3a", "3b", "3c", "3d", "3e", "3f", "3j", "3k", "3n", "3o"],
      "RQ4": ["3g", "3h", "3i", "3m"], "RQ5": ["4a", "4b"]}
print("rq,total,peer,share")
for q, s in RQ.items():
    sup = [r for r in rows if cats(r) & set(s)]
    peer = sum(r["Venue Type"] in ("conference", "journal") for r in sup)
    print(f"{q},{len(sup)},{peer},{peer/len(sup):.2f}")
import collections
cc = collections.Counter(c for r in peer_all for c in cats(r))
print("peer top categories", cc.most_common(6))
ca = collections.Counter(c for r in rows for c in cats(r))
print("all top categories", ca.most_common(6))
