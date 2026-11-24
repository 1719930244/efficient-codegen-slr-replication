#!/usr/bin/env python3
"""Check that every Stage-1 label follows from the released rule.

Reads stage1-labels.csv, applies classify.decide() to the matched-term counts
of each record (records before 2017 are excluded by the year cut-off), and
reports any record whose released label differs from the rule.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from classify import YEAR_CUTOFF, decide  # noqa: E402

COUNTS = ["ic_codegen", "ic_efficiency", "ic_llm", "ec_non_codegen", "ec_survey"]


def expected(row):
    year = row["year"]
    if year.isdigit() and int(year) < YEAR_CUTOFF:
        return "Likely-Exclude"
    return decide({k: int(row[k] or 0) for k in COUNTS})


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "stage1-labels.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    mismatches = [r["key"] for r in rows if expected(r) != r["label"]]
    print(f"records: {len(rows)}")
    for label, n in sorted(Counter(r["label"] for r in rows).items()):
        print(f"  {label:15s} {n:6d}")
    print(f"labels that differ from the rule: {len(mismatches)}")
    if mismatches:
        print("first differing keys:", ", ".join(mismatches[:10]))
        sys.exit(1)


if __name__ == "__main__":
    main()
