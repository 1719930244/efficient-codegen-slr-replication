#!/usr/bin/env python3
"""
Fine-grained classification of 123 primary studies.
Reads abstracts from bib/PDFs, uses LLM to assign RQ + sub-category labels.
"""

import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import openpyxl

BASE = Path("/home/szw/overleaf/efficient-codegen-survey/data")
PDF_DIR = BASE / "pdfs"
BIB_FILE = BASE / "02-dedup" / "deduplicated.bib"
XLSX_FILE = BASE / "primary-studies-assessment.xlsx"
OUTPUT_CSV = BASE / "fine-grained-classification.csv"
PROGRESS_FILE = BASE / "classification-progress.json"

# LLM API config (same as screening)
API_URL = "https://code.mmkg.cloud/v1/chat/completions"
API_KEY = os.environ.get("LLM_API_KEY", "")  # key removed from the public package; set LLM_API_KEY to re-run
MODEL = "gpt-5-mini"

# ── Fine-grained taxonomy ──────────────────────────────────────────────────

TAXONOMY = """
## RQ1: Data-Level Efficiency
- 1a-DataSelection: Data selection, filtering, or sampling strategies for training
- 1b-DataQuality: Data quality assessment, augmentation, or cleaning
- 1c-DataDedup: Data deduplication techniques
- 1d-DataMixing: Data mixing strategies (code vs NL ratio, domain mixing)

## RQ2: Training-Level Efficiency
- 2a-EfficientPretrain: Efficient pre-training (architecture design, training objectives, tokenizer optimization)
- 2b-PEFT: Parameter-efficient fine-tuning (LoRA, Adapter, Prefix Tuning, etc.)
- 2c-Distillation: Knowledge distillation (large-to-small model transfer)
- 2d-CurriculumLearning: Curriculum learning or self-paced learning for training
- 2e-TrainCompression: Training-time compression (QAT, structured pruning during training)
- 2f-RLTraining: Reinforcement learning for code generation training (RLHF, reward models)

## RQ3: Inference and Application-Level Efficiency
- 3a-SpeculativeDecoding: Speculative decoding, parallel decoding, draft-then-verify
- 3b-EarlyExit: Early exit, adaptive computation depth, excess token prevention
- 3c-NAR: Non-autoregressive or semi-autoregressive generation
- 3d-PromptCompression: Prompt/input compression, docstring compression, token-efficient representations
- 3e-ContextPruning: Context pruning, filtering, or RAG optimization for retrieval-augmented generation
- 3f-KVCache: KV cache optimization (prediction, eviction, anchor attention)
- 3g-PTQ: Post-training quantization
- 3h-ModelPruning: Post-training model pruning, low-rank decomposition, compression
- 3i-Routing: Model cascading, routing, dynamic model selection, MoE at inference
- 3j-Sampling: Efficient sampling strategies (adaptive consistency, compute allocation)
- 3k-MultiAgent: Multi-agent collaboration optimization (cost/token reduction in agent pipelines)
- 3l-CodeSpecific: Code-specific inference strategies (syntax-aware, incremental generation, grammar-based)
- 3m-SystemOpt: System-level optimization (serving, scheduling, batching, SLA-aware)
- 3n-PromptEng: Prompt engineering / prompt pattern optimization for cost reduction
- 3o-CoTOpt: Chain-of-thought optimization (compression, chain-of-draft)

## RQ4: Evaluation
- 4a-Benchmark: Proposing or analyzing efficiency benchmarks for code generation
- 4b-EmpiricalStudy: Empirical evaluation/comparison studies of efficiency techniques
- 4c-MetricsReporting: Analysis of metrics usage, reporting practices, or gaps

## Scope Flags
- SCOPE-GenEffCode: Paper primarily about generating EFFICIENT code (runtime performance of output), not about efficient generation process. May be out of scope.
- SCOPE-Borderline: Paper is borderline - partially about generation efficiency but main focus is elsewhere.
"""

SYSTEM_PROMPT = f"""You are a research paper classifier for a systematic literature review on "Efficiency in LLM-based Code Generation".

IMPORTANT SCOPE DEFINITION:
- IN SCOPE: Techniques that make the CODE GENERATION PROCESS more efficient (less latency, memory, cost, energy, tokens, training time)
- OUT OF SCOPE: Techniques that make the GENERATED CODE run faster/more efficiently (e.g., optimizing CUDA kernels for runtime speed, generating gas-efficient smart contracts)
- BORDERLINE: Papers that do both (e.g., generate efficient CUDA code AND report on generation cost/latency)

Given a paper's title, abstract, and screening reason, classify it into one or more fine-grained categories.

{TAXONOMY}

Rules:
1. Assign 1-3 primary categories (most papers have 1-2)
2. Categories can span multiple RQs (e.g., "2b-PEFT; 4b-EmpiricalStudy")
3. Add SCOPE flags if the paper's main contribution is about generated code efficiency, not generation efficiency
4. For papers that genuinely span data/training/inference, assign all relevant categories
5. Be conservative: only assign categories where the paper makes a clear contribution

Output format (JSON):
{{"primary_categories": ["3a-SpeculativeDecoding"], "secondary_categories": [], "scope_flags": [], "rq_assignment": "RQ3", "brief_rationale": "Proposes speculative decoding for code completion with 2.5x speedup"}}

For multi-RQ papers:
{{"primary_categories": ["2b-PEFT", "4b-EmpiricalStudy"], "secondary_categories": [], "scope_flags": [], "rq_assignment": "RQ2; RQ4", "brief_rationale": "Empirical study comparing PEFT methods for code models"}}

For scope-flagged papers:
{{"primary_categories": ["3k-MultiAgent"], "secondary_categories": [], "scope_flags": ["SCOPE-GenEffCode"], "rq_assignment": "RQ3", "brief_rationale": "Multi-agent CUDA kernel optimization - main focus is generating faster kernels, but reports generation cost"}}
"""


# ── Abstract extraction ────────────────────────────────────────────────────

def parse_bib_abstracts(bib_path):
    """Parse abstracts from BibTeX file."""
    with open(bib_path, "r", encoding="utf-8") as f:
        content = f.read()

    abstracts = {}
    # Match entries
    for m in re.finditer(r'@\w+\{([^,]+),\s*(.*?)(?=\n@|\Z)', content, re.DOTALL):
        key = m.group(1).strip()
        body = m.group(2)
        abs_match = re.search(
            r'abstract\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}',
            body, re.IGNORECASE | re.DOTALL
        )
        if abs_match:
            abstract = abs_match.group(1).strip()
            # Clean up LaTeX
            abstract = re.sub(r'\s+', ' ', abstract)
            abstracts[key] = abstract

    return abstracts


def extract_abstract_from_pdf(pdf_path):
    """Extract abstract from PDF using pdftotext or PyPDF2."""
    text = ""

    # Method 1: pdftotext (better quality)
    try:
        import subprocess
        result = subprocess.run(
            ["pdftotext", "-f", "1", "-l", "2", str(pdf_path), "-"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            text = result.stdout
    except Exception:
        pass

    # Method 2: PyPDF2 fallback
    if not text:
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(pdf_path))
            for page_num in range(min(2, len(reader.pages))):
                page_text = reader.pages[page_num].extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            print(f"  PDF extraction failed: {e}", file=sys.stderr)
            return ""

    if not text:
        return ""

    # Clean up
    full_text = ' '.join(text.split())

    # Look for "Abstract" marker
    abs_match = re.search(
        r'(?:Abstract|ABSTRACT)[.\s:—\-]*(.{100,2000}?)(?:Introduction|INTRODUCTION|1[\s.]?\s*Introduction|Keywords|Index Terms|CCS Concepts|I\.\s+INTRODUCTION)',
        full_text, re.DOTALL | re.IGNORECASE
    )
    if abs_match:
        return abs_match.group(1).strip()

    # Fallback: return first 1500 chars
    if len(full_text) > 200:
        return full_text[:1500].strip()
    return ""


def get_abstract(key, title, bib_abstracts):
    """Get abstract for a paper from bib or PDF."""
    # 1. Try bib
    if key in bib_abstracts:
        return bib_abstracts[key]

    # 2. Try PDF with various key formats
    pdf_candidates = [
        PDF_DIR / f"{key}.pdf",
    ]
    # Handle arxiv_ prefix
    if key.startswith("arxiv_"):
        alt_key = key.replace("arxiv_", "")
        pdf_candidates.append(PDF_DIR / f"{alt_key}.pdf")
    # Handle doi_ prefix -> try arxiv id extraction
    if key.startswith("doi_10_48550_arXiv_"):
        arxiv_id = key.replace("doi_10_48550_arXiv_", "")
        pdf_candidates.append(PDF_DIR / f"{arxiv_id}.pdf")

    for pdf_path in pdf_candidates:
        if pdf_path.exists():
            abstract = extract_abstract_from_pdf(pdf_path)
            if abstract:
                return abstract

    return ""


# ── LLM classification ────────────────────────────────────────────────────

def classify_paper(title, abstract, screening_reason, lifecycle_stage):
    """Call LLM to classify a paper."""
    import urllib.request

    user_msg = f"""Title: {title}

Abstract: {abstract[:2000] if abstract else '(No abstract available)'}

Screening reason: {screening_reason if screening_reason else '(Not available)'}

Current (auto) lifecycle stage: {lifecycle_stage}

Please classify this paper into fine-grained categories."""

    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0.1,
        "max_tokens": 500,
    })

    req = urllib.request.Request(
        API_URL,
        data=payload.encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            # Parse JSON from response
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                print(f"  Warning: no JSON found in response", file=sys.stderr)
                return {"error": "no JSON in response", "raw": content[:500]}
    except Exception as e:
        print(f"  API error: {e}", file=sys.stderr)
        return {"error": str(e)}


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")

    # Load xlsx
    wb = openpyxl.load_workbook(XLSX_FILE)
    ws = wb["Primary Studies"]
    papers = []
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, values_only=True):
        papers.append({
            "num": row[0],
            "key": row[1],
            "title": row[2],
            "author": row[3],
            "year": row[4],
            "venue": row[5],
            "venue_type": row[6],
            "source": row[7],
            "lifecycle_stage": row[19],
        })

    # Load screening reasons
    screening_reasons = {}
    screening_csv = BASE / "03-screening" / "screening-llm-merged.csv"
    with open(screening_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("decision") == "INCLUDE":
                screening_reasons[r["key"]] = r.get("reason", "")

    # Load bib abstracts
    print("Parsing bib abstracts...")
    bib_abstracts = parse_bib_abstracts(BIB_FILE)
    print(f"  Found {len(bib_abstracts)} abstracts in bib")

    # Load progress (resume support)
    progress = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r") as f:
            progress = json.load(f)
        print(f"  Resuming: {len(progress)} papers already classified")

    # Classify each paper
    results = []
    total = len(papers)

    for i, paper in enumerate(papers):
        key = paper["key"]
        title = paper["title"]

        if key in progress:
            results.append(progress[key])
            continue

        print(f"[{i+1}/{total}] {key}: {title[:60]}...")

        abstract = get_abstract(key, title, bib_abstracts)
        reason = screening_reasons.get(key, "")

        classification = classify_paper(
            title, abstract, reason, paper["lifecycle_stage"]
        )

        result = {
            "num": paper["num"],
            "key": key,
            "title": title,
            "year": paper["year"],
            "venue": paper["venue"],
            "old_lifecycle": paper["lifecycle_stage"],
            "has_abstract": bool(abstract),
            **classification,
        }
        results.append(result)
        progress[key] = result

        # Save progress after each paper
        with open(PROGRESS_FILE, "w") as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)

        # Rate limit
        time.sleep(0.3)

    # Write output CSV
    print(f"\nWriting {len(results)} results to {OUTPUT_CSV}...")

    fieldnames = [
        "num", "key", "title", "year", "venue",
        "old_lifecycle",
        "primary_categories", "secondary_categories", "scope_flags",
        "rq_assignment", "brief_rationale",
        "has_abstract",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = dict(r)
            # Convert lists to semicolon-separated strings
            for list_field in ["primary_categories", "secondary_categories", "scope_flags"]:
                if isinstance(row.get(list_field), list):
                    row[list_field] = "; ".join(row[list_field])
            writer.writerow(row)

    # Print summary
    print("\n=== Classification Summary ===")
    from collections import Counter
    rq_counts = Counter()
    cat_counts = Counter()
    scope_flags = Counter()
    for r in results:
        rq = r.get("rq_assignment", "UNKNOWN")
        for rq_part in rq.split(";"):
            rq_counts[rq_part.strip()] += 1
        for cat in r.get("primary_categories", []):
            if isinstance(cat, str):
                cat_counts[cat] += 1
        for flag in r.get("scope_flags", []):
            if isinstance(flag, str):
                scope_flags[flag] += 1

    print("\nRQ distribution:")
    for rq, count in sorted(rq_counts.items()):
        print(f"  {rq}: {count}")

    print("\nTop categories:")
    for cat, count in cat_counts.most_common(20):
        print(f"  {cat}: {count}")

    if scope_flags:
        print("\nScope flags:")
        for flag, count in scope_flags.most_common():
            print(f"  {flag}: {count}")

    print(f"\nDone! Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
