#!/usr/bin/env python3
"""Build the 200-record title-and-abstract dual-screening sheets.

The sample is drawn from the recovered Stage-1 auto-screening table:
100 records labeled Likely-Include or Uncertain, and 100 labeled Likely-Exclude.
The seed is fixed at 20261007. Abstracts are filled from the recovered
deduplicated bibliography and the abstract supplement when available; many
Stage-1 records were title-only, so some abstract cells are empty by design.
"""
import csv
import json
import random
import re
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Font

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "original-2026-03"
BIB = Path("/tmp/tosem-bib/02-dedup/deduplicated.bib")
SUP = Path("/tmp/tosem-bib/01-search/abstracts-supplemented.json")

rows = list(csv.DictReader(open(DATA / "03-screening/screening-ta-auto.csv", encoding="utf-8-sig")))
entries = {r["key"]: r for r in csv.DictReader(open(DATA / "entries-summary.csv", encoding="utf-8-sig"))}

abstracts = {}
if BIB.exists():
    text = BIB.read_text(encoding="utf-8", errors="replace")
    for m in re.finditer(r"@\w+\{([^,]+),(.*?)(?=\n@|\Z)", text, flags=re.S):
        a = re.search(r"abstract\s*=\s*\{(.*?)\}\s*,?\s*\n", m.group(2), flags=re.S | re.I)
        if a:
            abstracts[m.group(1).strip()] = re.sub(r"\s+", " ", a.group(1)).strip()
if SUP.exists():
    for key, value in json.load(open(SUP, encoding="utf-8")).items():
        if isinstance(value, dict) and value.get("abstract"):
            abstracts[key] = re.sub(r"\s+", " ", value["abstract"]).strip()

retained = [r for r in rows if r["label"] in ("Likely-Include", "Uncertain")]
excluded = [r for r in rows if r["label"] == "Likely-Exclude"]
rnd = random.Random(20261007)
sample = rnd.sample(retained, 100) + rnd.sample(excluded, 100)
rnd.shuffle(sample)

for rater in (1, 2):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "screening"
    ws.append(["ID", "Title", "Abstract", "Year", "Venue", "Decision (Include/Exclude/Uncertain)", "Criterion", "Note"])
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for row in sample:
        e = entries.get(row["key"], {})
        ws.append([row["key"], row["title"], abstracts.get(row["key"], ""), row.get("year", "") or e.get("year", ""), e.get("source_db", ""), "", "", ""])
    for col, width in zip("ABCDEFGH", (18, 50, 90, 8, 20, 28, 14, 30)):
        ws.column_dimensions[col].width = width
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    wb.save(ROOT / f"TA-screening-Rater{rater}.xlsx")

json.dump(
    [{"key": r["key"], "stratum": r["label"], "has_abstract": bool(abstracts.get(r["key"]))} for r in sample],
    open(ROOT / "sample-ids-ta.json", "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=1,
)
print(f"wrote {len(sample)} rows; abstracts available for {sum(1 for r in sample if abstracts.get(r['key']))}")