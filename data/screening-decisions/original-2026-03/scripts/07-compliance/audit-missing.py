#!/usr/bin/env python3
"""
Audit the 20 xlsx-in-scope papers that were not in final-compliance.json
to achieve full N=122 coverage. Merges results into final-compliance.json
using the schema already present in that file (fc/model/hw/lat/mem/serve/cost/energy).
"""
import json
import os
import re
import sys
import importlib.util

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
XLSX = os.path.join(DATA_DIR, "primary-studies-assessment.xlsx")
FINAL = os.path.join(SCRIPT_DIR, "final-compliance.json")

spec = importlib.util.spec_from_file_location(
    "ac", os.path.join(SCRIPT_DIR, "analyze-compliance.py")
)
ac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ac)

from openpyxl import load_workbook

DOI_TO_ARXIV = {
    "doi_10_1109_TCAD_2024_3483089": "2312_08617",   # RTLCoder
    "doi_10_1109_ASE56229_2023_00125": "2308_10462", # PEFT empirical study
    "doi_10_1109_TSE_2023_3267446": "2208_08227",    # MultiPL-E
}

FIELD_MAP = [
    ("functional_correctness", "fc"),
    ("model_info", "model"),
    ("hardware", "hw"),
    ("latency_throughput", "lat"),
    ("memory", "mem"),
    ("serving_config", "serve"),
    ("monetary_cost", "cost"),
    ("energy_carbon", "energy"),
]


def resolve_pdf(key):
    if key in DOI_TO_ARXIV:
        p = os.path.join(PDF_DIR, f"{DOI_TO_ARXIV[key]}.pdf")
        if os.path.exists(p):
            return p
    p = os.path.join(PDF_DIR, f"{key}.pdf")
    if os.path.exists(p):
        return p
    stripped = key
    for pref in ("arxiv_", "doi_"):
        if key.startswith(pref):
            stripped = key[len(pref):]
            break
    p = os.path.join(PDF_DIR, f"{stripped}.pdf")
    if os.path.exists(p):
        return p
    return None


def main():
    wb = load_workbook(XLSX)
    ws = wb.active
    headers = [c.value for c in ws[1]]
    key_col = headers.index("key")
    title_col = headers.index("title")
    rq_col = headers.index("new_rq")

    xlsx_papers = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        k = row[key_col]
        rq = row[rq_col]
        t = row[title_col]
        if rq and str(rq).strip():
            xlsx_papers[k] = {"title": t or "", "rq": str(rq).strip()}

    with open(FINAL) as f:
        final = json.load(f)

    # Filter out entries not in xlsx in-scope (these are papers removed from
    # the final primary study set but still present in the older CSV-driven run)
    orphans = [r["key"] for r in final["details"] if r["key"] not in xlsx_papers]
    final["details"] = [r for r in final["details"] if r["key"] in xlsx_papers]
    print(f"filtered out {len(orphans)} orphan(s) not in xlsx in-scope: {orphans}")

    existing = {r["key"] for r in final["details"]}
    missing = [k for k in xlsx_papers if k not in existing]
    print(f"xlsx in-scope: {len(xlsx_papers)}")
    print(f"already audited (post-filter): {len(existing)}")
    print(f"missing: {len(missing)}")

    new_results = []
    not_resolved = []
    errors = []
    for k in missing:
        pdf = resolve_pdf(k)
        if not pdf:
            not_resolved.append(k)
            print(f"  MISS PDF: {k}")
            continue
        r = ac.analyze_paper(k, pdf)
        if r is None:
            errors.append(k)
            print(f"  ERROR   : {k} ({pdf})")
            continue
        out = {"key": k, "title": xlsx_papers[k]["title"]}
        for long_f, short_f in FIELD_MAP:
            out[short_f] = bool(r[long_f])
        new_results.append(out)
        print(
            f"  OK {k:<38} fc={int(out['fc'])} mdl={int(out['model'])} "
            f"hw={int(out['hw'])} lat={int(out['lat'])} mem={int(out['mem'])} "
            f"srv={int(out['serve'])} $={int(out['cost'])} e={int(out['energy'])}"
        )

    final["details"].extend(new_results)
    final["analyzed"] = len(final["details"])
    final["total"] = len(xlsx_papers)
    final["missing_pdfs"] = len(not_resolved)
    if "errors" in final or errors:
        final["errors"] = errors

    # Recompute aggregate metrics using short field names
    metric_labels = {
        "Functional correctness (pass@k)": "fc",
        "Model name / version": "model",
        "Hardware specification": "hw",
        "Latency / throughput": "lat",
        "Memory consumption": "mem",
        "Serving configuration": "serve",
        "Monetary cost": "cost",
        "Energy / carbon": "energy",
    }
    n = final["analyzed"]
    final["metrics"] = {
        label: sum(1 for r in final["details"] if r.get(f, False))
        for label, f in metric_labels.items()
    }

    with open(FINAL, "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"\n=== Aggregate (N={n}) ===")
    for label, f in metric_labels.items():
        c = final["metrics"][label]
        print(f"  {label:<36} {c:>4} ({100*c/n:>5.1f}%)")

    # Per-RQ breakdown using xlsx rq mapping (so every row has one primary rq)
    rq_map = {k: v["rq"] for k, v in xlsx_papers.items()}
    rq_buckets = {}
    for r in final["details"]:
        rq_str = rq_map.get(r["key"], "")
        rqs = [x.strip() for x in re.split(r"[,;]", rq_str) if x.strip()]
        if not rqs:
            rqs = ["(none)"]
        for rq in rqs:
            rq_buckets.setdefault(rq, []).append(r)

    print(f"\n=== Per-RQ (studies may count in multiple RQs) ===")
    for rq in sorted(rq_buckets):
        rs = rq_buckets[rq]
        nr = len(rs)
        parts = [f"{rq:>8} N={nr:<3}"]
        for _, sf in FIELD_MAP:
            c = sum(1 for r in rs if r.get(sf, False))
            pct = 100 * c / nr if nr else 0
            parts.append(f"{sf}={pct:>4.0f}%")
        print("  ".join(parts))

    if not_resolved:
        print(f"\nSTILL MISSING PDFs: {not_resolved}")
    if errors:
        print(f"EXTRACTION ERRORS: {errors}")


if __name__ == "__main__":
    main()
