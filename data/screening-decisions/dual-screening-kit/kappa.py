#!/usr/bin/env python3
"""Compute Cohen's kappa and raw agreement for a dual-screening sheet pair.

Usage:
  python kappa.py TA-screening-Rater1.xlsx TA-screening-Rater2.xlsx
  python kappa.py FT-screening-Rater1.xlsx FT-screening-Rater2.xlsx

The decision column is located by its header, so the script works for both the
title-and-abstract kit and the full-text kit. Empty decisions are ignored; the
two sheets must cover the same IDs.
"""
import sys
import openpyxl


def load(path):
    ws = openpyxl.load_workbook(path, data_only=True).active
    rows = list(ws.iter_rows(values_only=True))
    header = [str(x or "").strip() for x in rows[0]]
    try:
        id_col = next(i for i, h in enumerate(header) if h.lower() == "id")
        dec_col = next(i for i, h in enumerate(header) if "decision" in h.lower())
    except StopIteration:
        raise SystemExit("Could not find ID and Decision columns in " + path)
    out = {}
    for r in rows[1:]:
        if len(r) <= max(id_col, dec_col):
            continue
        key = r[id_col]
        dec = str(r[dec_col] or "").strip().capitalize()
        if key and dec:
            out[str(key)] = dec
    return out


def kappa(x, y):
    n = len(x)
    if n == 0:
        return 0.0, 0.0
    po = sum(p == q for p, q in zip(x, y)) / n
    cats = sorted(set(x) | set(y))
    pe = sum((x.count(c) / n) * (y.count(c) / n) for c in cats)
    return po, ((po - pe) / (1 - pe)) if pe < 1 else 1.0


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python kappa.py Rater1.xlsx Rater2.xlsx")
    a, b = load(sys.argv[1]), load(sys.argv[2])
    ids = [i for i in a if i in b]
    x = [a[i] for i in ids]
    y = [b[i] for i in ids]
    po, k = kappa(x, y)
    print(f"n={len(ids)}  three-class: agreement={po:.3f} kappa={k:.3f}")
    bx = ["Exclude" if v == "Exclude" else "Include" for v in x]
    by = ["Exclude" if v == "Exclude" else "Include" for v in y]
    po, k = kappa(bx, by)
    print(f"n={len(ids)}  binary (Uncertain->Include): agreement={po:.3f} kappa={k:.3f}")
    print("disagreements:", [i for i in ids if a[i] != b[i]])


if __name__ == "__main__":
    main()