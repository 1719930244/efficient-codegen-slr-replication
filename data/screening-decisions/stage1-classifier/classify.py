#!/usr/bin/env python3
"""Reconstruction of the Stage-1 keyword classifier described in the manuscript.

This is a faithful implementation of the published scoring rule, not the
original executed script. It reads records with title and abstract fields,
scores them against the Group A/B/C vocabularies, and writes a ranked CSV.

The executed threshold and per-group weights are not stored in this package.
Pass them explicitly with --threshold and --weights; do not assume the values
used in the manuscript unless the authors confirm them.

Usage:
  python classify.py --input records.jsonl --output ranked.csv \
      --threshold 0.0 --weights 1,1,1

Input: JSONL or CSV with fields id, title, abstract (missing abstract is "").
Output columns: id, title, score_a, score_b, score_c, score, retained.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

GROUP_A = [
    "code generation", "code completion", "code synthesis",
    "program synthesis", "code infilling", "automated programming",
]
GROUP_B = [
    "efficien", "optimi", "accelerat", "lightweight", "compress",
    "quantiz", "prun", "distill", "latency", "throughput",
    "computational cost", "energy", "scalab",
]
GROUP_C = [
    "large language model", "llm", "language model", "transformer",
    "pre-trained model", "pretrained model", "foundation model",
    "neural network", "deep learning",
]


def matches(text, terms):
    low = text.lower()
    hits = 0
    for term in terms:
        # Prefix match at a word boundary, so plurals and inflections count.
        base = term[:-1] if term.endswith("*") else term
        if re.search(r"\b" + re.escape(base), low):
            hits += 1
    return hits


def read_records(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                yield row
    else:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--weights", default="1,1,1",
                    help="comma-separated weights for Groups A,B,C")
    args = ap.parse_args()
    weights = [float(x) for x in args.weights.split(",")]
    if len(weights) != 3:
        sys.exit("--weights must have three comma-separated values")

    rows = []
    for rec in read_records(args.input):
        title = str(rec.get("title") or "")
        abstract = str(rec.get("abstract") or "")
        text = title + " " + abstract
        a = matches(text, GROUP_A) * weights[0]
        b = matches(text, GROUP_B) * weights[1]
        c = matches(text, GROUP_C) * weights[2]
        # The published rule zeroes the score when any group is unmatched.
        score = 0.0 if min(a, b, c) == 0 else a + b + c
        rows.append({
            "id": rec.get("id") or rec.get("key") or rec.get("openalex_id") or "",
            "title": title,
            "score_a": a,
            "score_b": b,
            "score_c": c,
            "score": score,
            "retained": "yes" if score >= args.threshold else "no",
        })

    rows.sort(key=lambda r: (-r["score"], r["title"]))
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "title", "score_a", "score_b", "score_c", "score", "retained"])
        w.writeheader()
        w.writerows(rows)
    print(f"scored {len(rows)} records; retained {sum(1 for r in rows if r['retained']=='yes')}")


if __name__ == "__main__":
    main()