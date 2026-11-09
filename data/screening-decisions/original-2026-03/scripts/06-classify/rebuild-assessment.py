#!/usr/bin/env python3
"""
Rebuild assessment xlsx after removing 16 papers (14 scope + 2 duplicate).
Adds fine-grained classification columns.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

BASE = Path("/home/szw/overleaf/efficient-codegen-survey/data")

# Load removal list
with open(BASE / "removed-papers.json") as f:
    removed = json.load(f)
REMOVE_KEYS = set(removed["scope_geneffcode"] + removed["duplicate"])

# Load fine-grained classification
classification = {}
with open(BASE / "fine-grained-classification.csv", "r") as f:
    for row in csv.DictReader(f):
        classification[row["key"]] = row

# Load original xlsx
wb_old = openpyxl.load_workbook(BASE / "primary-studies-assessment.xlsx")
ws_old = wb_old["Primary Studies"]

# Read all data
headers_old = [cell.value for cell in ws_old[1]]
papers = []
removed_papers = []
for row in ws_old.iter_rows(min_row=2, max_row=ws_old.max_row, values_only=True):
    data = dict(zip(headers_old, row))
    key = data["key"]
    if key in REMOVE_KEYS:
        removed_papers.append(data)
        continue
    # Add classification data
    cls = classification.get(key, {})
    data["new_rq"] = cls.get("new_rq", data.get("lifecycle_stage", ""))
    data["primary_categories"] = cls.get("primary_categories", "")
    data["secondary_categories"] = cls.get("secondary_categories", "")
    data["scope_flags"] = cls.get("scope_flags", "")
    data["brief_rationale"] = cls.get("brief_rationale", "")
    papers.append(data)

print(f"Original: {ws_old.max_row - 1} papers")
print(f"Removed: {len(removed_papers)}")
print(f"Remaining: {len(papers)}")

# Renumber
for i, p in enumerate(papers, 1):
    p["#"] = i

# ── Build new workbook ──────────────────────────────────────────────────────

wb = openpyxl.Workbook()

# Styles
header_font = Font(bold=True, color="FFFFFF", size=11)
header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
thin_border = Border(
    left=Side(style="thin", color="D9D9D9"),
    right=Side(style="thin", color="D9D9D9"),
    top=Side(style="thin", color="D9D9D9"),
    bottom=Side(style="thin", color="D9D9D9"),
)
white_fill = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")
gray_fill = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
red_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
center_align = Alignment(horizontal="center", vertical="center")

def style_header(ws, ncols):
    for col in range(1, ncols + 1):
        cell = ws.cell(row=1, column=col)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_alignment
        cell.border = thin_border

def style_data_rows(ws, nrows, ncols):
    for row in range(2, nrows + 2):
        fill = white_fill if (row % 2 == 0) else gray_fill
        for col in range(1, ncols + 1):
            cell = ws.cell(row=row, column=col)
            cell.fill = fill
            cell.border = thin_border

# ── Sheet 1: Primary Studies ────────────────────────────────────────────────

ws1 = wb.active
ws1.title = "Primary Studies"

new_headers = [
    "#", "key", "title", "author", "year", "venue", "venue_type", "source",
    "IC1", "IC2", "IC3", "IC4", "EC_triggered", "inclusion_reason",
    "QA1", "QA2", "QA3", "QA4", "QA_total",
    "new_rq", "primary_categories", "secondary_categories",
]
ws1.append(new_headers)

for p in papers:
    ws1.append([
        p["#"], p["key"], p["title"], p["author"], p["year"],
        p["venue"], p["venue_type"], p["source"],
        p["IC1"], p["IC2"], p["IC3"], p["IC4"],
        p["EC_triggered"], p["inclusion_reason"],
        p["QA1"], p["QA2"], p["QA3"], p["QA4"], p["QA_total"],
        p["new_rq"], p["primary_categories"], p["secondary_categories"],
    ])

ncols1 = len(new_headers)
style_header(ws1, ncols1)
style_data_rows(ws1, len(papers), ncols1)

col_widths1 = [5, 22, 55, 22, 6, 30, 12, 18,
               30, 30, 30, 30, 10, 50,
               6, 6, 6, 6, 8,
               20, 35, 25]
for i, w in enumerate(col_widths1, 1):
    ws1.column_dimensions[get_column_letter(i)].width = w

ws1.freeze_panes = "A2"

for row in range(2, len(papers) + 2):
    for col in [1, 5, 7, 8, 15, 16, 17, 18, 19]:
        ws1.cell(row=row, column=col).alignment = center_align

# ── Sheet 2: Summary ────────────────────────────────────────────────────────

ws2 = wb.create_sheet("Summary")

def write_section(ws, row, title, font_size=12):
    cell = ws.cell(row=row, column=1, value=title)
    cell.font = Font(bold=True, size=font_size, color="4472C4")
    return row + 1

def write_kv(ws, row, key, value):
    ws.cell(row=row, column=1, value=key)
    ws.cell(row=row, column=2, value=value)
    ws.cell(row=row, column=1).font = Font(bold=True)
    return row + 1

row = 1
row = write_section(ws2, row, "Primary Studies Assessment Summary (Post-Cleanup)", 14)
row += 1
row = write_section(ws2, row, "Overview")
row = write_kv(ws2, row, "Total studies", len(papers))
row = write_kv(ws2, row, "Database screening", sum(1 for p in papers if p["source"] == "database"))
sb_back = sum(1 for p in papers if p["source"] == "snowball-backward")
sb_fwd = sum(1 for p in papers if p["source"] == "snowball-forward")
row = write_kv(ws2, row, "Snowball (backward)", sb_back)
row = write_kv(ws2, row, "Snowball (forward)", sb_fwd)
row = write_kv(ws2, row, "Total snowball", sb_back + sb_fwd)
row = write_kv(ws2, row, "Removed (scope)", len(removed["scope_geneffcode"]))
row = write_kv(ws2, row, "Removed (duplicate)", len(removed["duplicate"]))
row += 1

# By year
row = write_section(ws2, row, "Distribution by Year")
year_counts = Counter(p["year"] for p in papers)
for y in sorted(year_counts.keys()):
    row = write_kv(ws2, row, str(y), year_counts[y])
row += 1

# By venue type
row = write_section(ws2, row, "Distribution by Venue Type")
vtype_counts = Counter(p["venue_type"] for p in papers)
for vt in ["conference", "journal", "preprint"]:
    row = write_kv(ws2, row, vt, vtype_counts.get(vt, 0))
row += 1

# By RQ (new)
row = write_section(ws2, row, "Distribution by RQ (Fine-grained)")
rq_counts = Counter()
for p in papers:
    for rq_part in p["new_rq"].split(";"):
        rq_part = rq_part.strip()
        if rq_part:
            rq_counts[rq_part] += 1
for rq in sorted(rq_counts.keys()):
    row = write_kv(ws2, row, rq, rq_counts[rq])
row += 1

# By primary category
row = write_section(ws2, row, "Distribution by Primary Category")
cat_counts = Counter()
for p in papers:
    for cat in p["primary_categories"].split(";"):
        cat = cat.strip()
        if cat:
            cat_counts[cat] += 1
for cat, count in cat_counts.most_common():
    row = write_kv(ws2, row, cat, count)
row += 1

# QA score distribution
row = write_section(ws2, row, "QA Score Distribution")
qa_dist = Counter(p["QA_total"] for p in papers)
for score in sorted(qa_dist.keys()):
    row = write_kv(ws2, row, f"QA = {score}", qa_dist[score])
row += 1

avg_qa = sum(p["QA_total"] for p in papers) / len(papers) if papers else 0
row = write_kv(ws2, row, "Average QA score", round(avg_qa, 2))
row = write_kv(ws2, row, "Min QA score", min(p["QA_total"] for p in papers))
row = write_kv(ws2, row, "Max QA score", max(p["QA_total"] for p in papers))

ws2.column_dimensions["A"].width = 30
ws2.column_dimensions["B"].width = 15

# ── Sheet 3: Removed Papers ─────────────────────────────────────────────────

ws3 = wb.create_sheet("Removed Papers")
ws3.append(["#", "key", "title", "year", "venue", "removal_reason", "brief_rationale"])
for i, p in enumerate(removed_papers, 1):
    key = p["key"]
    cls = classification.get(key, {})
    if key in set(removed["duplicate"]):
        reason = "Duplicate"
    else:
        reason = "SCOPE-GenEffCode (focus on generated code efficiency, not generation efficiency)"
    ws3.append([
        i, key, p["title"], p["year"], p["venue"],
        reason, cls.get("brief_rationale", ""),
    ])

style_header(ws3, 7)
style_data_rows(ws3, len(removed_papers), 7)
ws3.column_dimensions["A"].width = 5
ws3.column_dimensions["B"].width = 30
ws3.column_dimensions["C"].width = 70
ws3.column_dimensions["D"].width = 6
ws3.column_dimensions["E"].width = 30
ws3.column_dimensions["F"].width = 40
ws3.column_dimensions["G"].width = 60
ws3.freeze_panes = "A2"

# ── Sheet 4: Snowball Papers ────────────────────────────────────────────────

# Read snowball data from original
ws_sb_old = wb_old["Snowball Papers"]
headers_sb = [cell.value for cell in ws_sb_old[1]]

ws4 = wb.create_sheet("Snowball Papers")
sb_papers = [p for p in papers if p["source"] != "database"]

new_sb_headers = headers_sb[:20] + ["new_rq", "primary_categories"] + headers_sb[20:]
ws4.append(new_sb_headers)

# Read original snowball data for source_paper / direction
sb_orig = {}
for row in ws_sb_old.iter_rows(min_row=2, max_row=ws_sb_old.max_row, values_only=True):
    sb_orig[row[1]] = dict(zip(headers_sb, row))

for i, p in enumerate(sb_papers, 1):
    key = p["key"]
    sb_data = sb_orig.get(key, {})
    ws4.append([
        i, key, p["title"], p["author"], p["year"],
        p["venue"], p["venue_type"], p["source"],
        p["IC1"], p["IC2"], p["IC3"], p["IC4"],
        p["EC_triggered"], p["inclusion_reason"],
        p["QA1"], p["QA2"], p["QA3"], p["QA4"], p["QA_total"],
        p.get("lifecycle_stage", ""),
        p["new_rq"], p["primary_categories"],
        sb_data.get("source_paper", ""), sb_data.get("direction", ""),
    ])

style_header(ws4, len(new_sb_headers))
style_data_rows(ws4, len(sb_papers), len(new_sb_headers))
ws4.freeze_panes = "A2"

# ── Save ────────────────────────────────────────────────────────────────────

output = BASE / "primary-studies-assessment.xlsx"
wb.save(output)
print(f"\nSaved to: {output}")
print(f"Final count: {len(papers)} primary studies")

# Print new year distribution
print("\n=== New Year Distribution ===")
for y in sorted(year_counts.keys()):
    print(f"  {y}: {year_counts[y]}")

# Print new RQ distribution
print("\n=== New RQ Distribution ===")
for rq in sorted(rq_counts.keys()):
    print(f"  {rq}: {rq_counts[rq]}")
