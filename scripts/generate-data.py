#!/usr/bin/env python3
"""
Generate replication package data from primary study sources.
Merges data from primary-studies-assessment.xlsx and fine-grained-classification.csv.
"""

import csv
import json
import os
import sys

try:
    import openpyxl
except ImportError:
    print("pip install openpyxl")
    sys.exit(1)

SURVEY_DIR = os.path.expanduser("~/overleaf/efficient-codegen-survey")
DATA_DIR = os.path.join(SURVEY_DIR, "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Category full names
CATEGORY_NAMES = {
    "1a": "Data Selection",
    "1b": "Data Quality",
    "1c": "Data Deduplication",
    "1d": "Data Mixing",
    "2a": "Efficient Pre-training",
    "2b": "Parameter-Efficient Fine-Tuning (PEFT)",
    "2c": "Knowledge Distillation",
    "2d": "Curriculum Learning",
    "2e": "Training-time Compression",
    "2f": "RL-based Training",
    "3a": "Speculative Decoding",
    "3b": "Early Exit",
    "3c": "Non-Autoregressive Generation",
    "3d": "Prompt Compression",
    "3e": "Context Pruning",
    "3f": "KV Cache Optimization",
    "3g": "Post-Training Quantization",
    "3h": "Model Pruning",
    "3i": "Model Routing",
    "3j": "Adaptive Sampling",
    "3k": "Multi-Agent Orchestration",
    "3l": "Code-Specific Optimization",
    "3m": "System-Level Serving",
    "3n": "Prompt Engineering",
    "3o": "Chain-of-Thought Optimization",
    "4a": "Benchmark Design",
    "4b": "Empirical Study",
}

RQ_NAMES = {
    "RQ1": "Data Preparation",
    "RQ2": "Model Training",
    "RQ3": "Inference Optimization",
    "RQ4": "Evaluation",
}


def load_xlsx():
    """Load primary studies from assessment xlsx."""
    wb = openpyxl.load_workbook(os.path.join(DATA_DIR, "primary-studies-assessment.xlsx"))
    ws = wb["Primary Studies"]
    headers = [cell.value for cell in ws[1]]
    studies = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row[1]:
            continue
        key = row[1]
        studies[key] = {
            "key": key,
            "title": row[2],
            "author": row[3],
            "year": row[4],
            "venue": row[5],
            "venue_type": row[6],
            "source": row[7],
            "rq": row[19],
            "primary_categories": row[20],
            "secondary_categories": row[21] if row[21] and row[21] != "None" else "",
            "qa1": row[14],
            "qa2": row[15],
            "qa3": row[16],
            "qa4": row[17],
            "qa_total": row[18],
        }
    return studies


def load_csv_extra():
    """Load papers from classification CSV that are not in xlsx."""
    with open(os.path.join(DATA_DIR, "fine-grained-classification.csv")) as f:
        rows = list(csv.DictReader(f))

    # Only included papers (no SCOPE-GenEffCode or DUPLICATE flags)
    included = [
        r for r in rows
        if "SCOPE-GenEffCode" not in r.get("scope_flags", "")
        and "DUPLICATE" not in r.get("scope_flags", "")
    ]
    return {r["key"]: r for r in included}


def merge_studies():
    """Merge xlsx and csv data to get complete primary study set."""
    xlsx_data = load_xlsx()
    csv_data = load_csv_extra()

    # Start with xlsx data
    all_studies = dict(xlsx_data)

    # Add papers from CSV that are not in xlsx
    for key, row in csv_data.items():
        if key not in all_studies:
            all_studies[key] = {
                "key": key,
                "title": row["title"],
                "author": "",
                "year": int(row["year"]),
                "venue": row["venue"],
                "venue_type": "",
                "source": "supplementary",
                "rq": row["new_rq"],
                "primary_categories": row["primary_categories"],
                "secondary_categories": row.get("secondary_categories", ""),
                "qa1": "",
                "qa2": "",
                "qa3": "",
                "qa4": "",
                "qa_total": "",
            }

    return all_studies


def parse_rqs(rq_str):
    """Parse RQ string like 'RQ2; RQ3' into list ['RQ2', 'RQ3']."""
    if not rq_str:
        return []
    return [r.strip() for r in rq_str.split(";")]


def parse_categories(cat_str):
    """Parse category string like '3a-SpeculativeDecoding; 3d-PromptCompression'."""
    if not cat_str or cat_str == "None":
        return []
    return [c.strip() for c in cat_str.split(";")]


def get_category_code(cat):
    """Extract code from category like '3a-SpeculativeDecoding' -> '3a'."""
    return cat.split("-")[0] if "-" in cat else cat


def write_primary_studies_csv(studies):
    """Write primary-studies.csv with all studies."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outpath = os.path.join(OUTPUT_DIR, "primary-studies.csv")

    # Sort by year then key
    sorted_studies = sorted(studies.values(), key=lambda s: (s["year"], s["key"]))

    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ID", "Key", "Title", "Year", "Venue", "Venue Type", "Source",
            "RQ", "Primary Categories", "Secondary Categories",
            "QA Total"
        ])
        for i, s in enumerate(sorted_studies, 1):
            writer.writerow([
                f"S{i:03d}",
                s["key"],
                s["title"],
                s["year"],
                s["venue"],
                s["venue_type"],
                s["source"],
                s["rq"],
                s["primary_categories"],
                s["secondary_categories"],
                s["qa_total"] if s["qa_total"] else "",
            ])

    print(f"Written {len(sorted_studies)} studies to {outpath}")
    return sorted_studies


def write_by_rq(studies):
    """Write per-RQ study lists."""
    rq_dir = os.path.join(OUTPUT_DIR, "by-rq")
    os.makedirs(rq_dir, exist_ok=True)

    rq_studies = {"RQ1": [], "RQ2": [], "RQ3": [], "RQ4": []}

    for s in studies:
        rqs = parse_rqs(s["rq"])
        for rq in rqs:
            if rq in rq_studies:
                rq_studies[rq].append(s)

    for rq, items in rq_studies.items():
        outpath = os.path.join(rq_dir, f"{rq.lower()}-studies.csv")
        with open(outpath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Key", "Title", "Year", "Venue", "Categories"])
            for s in sorted(items, key=lambda x: (x["year"], x["key"])):
                writer.writerow([
                    s["key"],
                    s["title"],
                    s["year"],
                    s["venue"],
                    s["primary_categories"],
                ])
        print(f"  {rq}: {len(items)} studies -> {outpath}")


def write_classification_scheme(studies):
    """Write classification-scheme.csv mapping categories to studies."""
    outpath = os.path.join(OUTPUT_DIR, "classification-scheme.csv")

    cat_studies = {}
    for s in studies:
        cats = parse_categories(s["primary_categories"])
        for cat in cats:
            code = get_category_code(cat)
            if code not in cat_studies:
                cat_studies[code] = []
            cat_studies[code].append(s["key"])

        sec_cats = parse_categories(s.get("secondary_categories", ""))
        for cat in sec_cats:
            code = get_category_code(cat)
            if code not in cat_studies:
                cat_studies[code] = []
            if s["key"] not in cat_studies[code]:
                cat_studies[code].append(s["key"])

    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Code", "Category", "RQ", "Study Count", "Study Keys"])
        for code in sorted(cat_studies.keys()):
            rq_num = code[0]
            rq = f"RQ{rq_num}"
            name = CATEGORY_NAMES.get(code, code)
            keys = sorted(cat_studies[code])
            writer.writerow([code, name, rq, len(keys), "; ".join(keys)])

    print(f"Written classification scheme to {outpath}")


def write_statistics(studies):
    """Write summary statistics as JSON."""
    from collections import Counter

    stats = {
        "total_studies": len(studies),
        "year_distribution": dict(Counter(s["year"] for s in studies)),
        "source_distribution": dict(Counter(s["source"] for s in studies if s["source"])),
        "venue_type_distribution": dict(Counter(s["venue_type"] for s in studies if s["venue_type"])),
        "rq_distribution": {},
        "category_distribution": {},
    }

    # RQ distribution
    rq_counts = Counter()
    for s in studies:
        for rq in parse_rqs(s["rq"]):
            rq_counts[rq] += 1
    stats["rq_distribution"] = dict(rq_counts)

    # Category distribution
    cat_counts = Counter()
    for s in studies:
        for cat in parse_categories(s["primary_categories"]):
            code = get_category_code(cat)
            cat_counts[code] += 1
    stats["category_distribution"] = dict(sorted(cat_counts.items()))

    outpath = os.path.join(OUTPUT_DIR, "statistics.json")
    with open(outpath, "w") as f:
        json.dump(stats, f, indent=2, sort_keys=False)
    print(f"Written statistics to {outpath}")


def main():
    all_studies = merge_studies()
    print(f"Total primary studies: {len(all_studies)}")

    sorted_studies = write_primary_studies_csv(all_studies)
    write_by_rq(sorted_studies)
    write_classification_scheme(sorted_studies)
    write_statistics(sorted_studies)


if __name__ == "__main__":
    main()
