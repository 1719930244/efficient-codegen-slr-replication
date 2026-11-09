#!/usr/bin/env python3
"""
Full-text IC/EC screening for SLR.
Reads PDFs, extracts text, evaluates inclusion/exclusion criteria.
Saves progress incrementally to screening-ft.csv.

IC/EC criteria (from sec-methodology.tex):
  IC1: Proposes or evaluates a technique that improves efficiency of code generation
       (data, training, or inference stage)
  IC2: LLM-based or Transformer-based approach
  IC3: Addresses code completion or NL-to-code generation tasks
  IC4: At least one quantitative efficiency metric (latency, memory, throughput,
       FLOPs, energy, cost, training time)
  EC1: Core technique does not target efficiency
  EC2: Targets SE tasks other than code generation
  EC3: Not written in English
  EC4: Survey, editorial, or short abstract (<4 pages)
  EC5: Conference+journal duplicate (handled separately)
"""

import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(SCRIPT_DIR, "pdfs")
TA_FILE = os.path.join(SCRIPT_DIR, "screening-ta-auto.csv")
CANDIDATES_FILE = os.path.join(SCRIPT_DIR, "candidates-planb.csv")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "screening-ft.csv")
PROGRESS_FILE = os.path.join(SCRIPT_DIR, "screening-ft-progress.json")


def safe_filename(key):
    return re.sub(r'[^\w\-.]', '_', key) + '.pdf'


def extract_text(pdf_path, max_pages=6):
    """Extract text from PDF using pdftotext. Returns (text, page_count)."""
    try:
        # Get page count
        result = subprocess.run(
            ["pdftotext", "-l", str(max_pages), pdf_path, "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30
        )
        text = result.stdout.decode('utf-8', errors='replace').strip()

        # Get total page count
        result2 = subprocess.run(
            ["pdfinfo", pdf_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10
        )
        pages = 0
        for line in result2.stdout.decode('utf-8', errors='replace').split('\n'):
            if line.startswith('Pages:'):
                pages = int(line.split(':')[1].strip())
                break

        return text, pages
    except Exception as e:
        return "", 0


def check_ic1_efficiency_for_codegen(text):
    """IC1: Proposes or evaluates efficiency technique for code generation."""
    text_lower = text.lower()

    # Code generation keywords
    codegen_patterns = [
        r'code\s+generation', r'code\s+completion', r'code\s+synthesis',
        r'program\s+synthesis', r'code\s+infilling', r'automated\s+programming',
        r'nl.to.code', r'text.to.code', r'natural\s+language\s+to\s+code',
        r'code\s+generat', r'generating\s+code', r'generate\s+code',
        r'fill.in.the.middle', r'code\s+suggestion', r'code\s+predict',
    ]
    codegen_score = sum(1 for p in codegen_patterns if re.search(p, text_lower))

    # Efficiency technique keywords
    eff_patterns = [
        r'efficien', r'optimi[zs]', r'accelerat', r'lightweight',
        r'compress', r'quantiz', r'prun[ie]', r'distill',
        r'latency', r'throughput', r'flops?', r'speed\s*up',
        r'inference\s+(?:time|speed|cost)', r'training\s+(?:time|cost|efficien)',
        r'memory\s+(?:footprint|usage|consumption|efficien)',
        r'computational\s+cost', r'energy\s+(?:consumption|efficien)',
        r'token.per.second', r'parameter.efficien', r'resource.efficien',
        r'model\s+size', r'smaller\s+model', r'compact\s+model',
        r'speculative\s+decod', r'kv.cache', r'batch(?:ing)?',
        r'lora', r'adapter', r'prefix.tun', r'prompt.tun',
    ]
    eff_score = sum(1 for p in eff_patterns if re.search(p, text_lower))

    return codegen_score, eff_score, codegen_score >= 2 and eff_score >= 3


def check_ic2_llm_based(text):
    """IC2: LLM-based or Transformer-based approach."""
    text_lower = text.lower()
    patterns = [
        r'large\s+language\s+model', r'\bllm\b', r'\bllms\b',
        r'transformer', r'pre.?train', r'fine.?tun',
        r'foundation\s+model', r'language\s+model',
        r'\bgpt\b', r'codex', r'copilot', r'code\s*llama', r'starcoder',
        r'deepseek', r'codegen\b', r'incoder', r'santacoder',
        r'wizardcoder', r'phind', r'magicoder', r'qwen',
        r'claude', r'gemini', r'palm', r'llama',
        r'attention\s+mechanism', r'self.attention',
        r'neural\s+(?:code|program)', r'deep\s+learning',
    ]
    score = sum(1 for p in patterns if re.search(p, text_lower))
    return score, score >= 2


def check_ic3_codegen_task(text):
    """IC3: Addresses code completion or NL-to-code generation tasks."""
    text_lower = text.lower()
    patterns = [
        r'code\s+completion', r'code\s+generation', r'program\s+synthesis',
        r'code\s+synthesis', r'nl.to.code', r'text.to.code',
        r'code\s+infilling', r'fill.in.the.middle',
        r'automated\s+programming', r'code\s+suggestion',
        r'humaneval', r'mbpp', r'apps\b', r'codecontests',
        r'bigcodebench', r'livecodebench', r'evalplus',
        r'pass@k', r'pass@\d',
    ]
    score = sum(1 for p in patterns if re.search(p, text_lower))
    return score, score >= 2


def check_ic4_efficiency_metric(text):
    """IC4: Reports at least one quantitative efficiency metric."""
    text_lower = text.lower()
    patterns = [
        r'latency', r'throughput', r'tokens?\s*/\s*s(?:ec)?',
        r'flops?', r'gflops?', r'tflops?',
        r'memory\s+(?:usage|footprint|consumption)',
        r'gpu\s+(?:memory|hours?|utiliz)',
        r'energy\s+consumption', r'carbon', r'co2',
        r'inference\s+time', r'decoding\s+time', r'generation\s+time',
        r'training\s+time', r'training\s+cost', r'wall.clock',
        r'speed\s*up', r'(?:\d+\.?\d*)\s*x\s+(?:fast|speed)',
        r'tokens?\s+per\s+second', r'api\s+cost', r'\$\s*/\s*(?:token|query|task)',
        r'parameter\s+count', r'model\s+size',
        r'compression\s+ratio', r'bit(?:s)?(?:\s+|-)per(?:\s+|-)(?:weight|param)',
    ]
    score = sum(1 for p in patterns if re.search(p, text_lower))
    return score, score >= 2


def check_ec1_not_efficiency_focused(text, ic1_eff_score):
    """EC1: Core technique does not target efficiency."""
    return ic1_eff_score < 2


def check_ec2_other_se_task(text):
    """EC2: Targets SE tasks other than code generation."""
    text_lower = text.lower()

    other_tasks = [
        r'defect\s+(?:detect|predict)', r'bug\s+(?:detect|predict|localiz)',
        r'vulnerability\s+detect', r'malware\s+detect',
        r'test\s+(?:generat|case)', r'unit\s+test',
        r'code\s+review', r'code\s+summar', r'code\s+search',
        r'code\s+clone', r'code\s+smell', r'code\s+translat',
        r'program\s+repair', r'automated\s+(?:program\s+)?repair',
        r'fault\s+localiz', r'software\s+maintenance',
        r'requirement\s+(?:engineer|analys)', r'software\s+design',
        r'log\s+(?:analys|pars)', r'incident',
    ]
    other_score = sum(1 for p in other_tasks if re.search(p, text_lower))

    # Check if the paper is primarily about another SE task
    # by comparing other_task mentions vs code generation mentions
    codegen_patterns = [
        r'code\s+generation', r'code\s+completion', r'code\s+synthesis',
        r'program\s+synthesis', r'code\s+infilling',
    ]
    codegen_score = sum(1 for p in codegen_patterns if re.search(p, text_lower))

    # If other SE task is dominant and code generation is minimal
    return other_score, other_score >= 3 and codegen_score <= 1


def check_ec3_not_english(text):
    """EC3: Not written in English."""
    if not text or len(text) < 100:
        return False
    # Check ratio of ASCII characters
    ascii_count = sum(1 for c in text[:2000] if ord(c) < 128)
    ratio = ascii_count / min(len(text), 2000)
    return ratio < 0.7


def check_ec4_survey_or_short(text, page_count, title):
    """EC4: Survey, editorial, or short abstract (<4 pages)."""
    title_lower = (title or '').lower()
    text_lower = text[:3000].lower() if text else ''

    is_survey = bool(re.search(
        r'\b(?:survey|systematic\s+(?:literature\s+)?review|tutorial|'
        r'literature\s+review|sota|state.of.the.art\s+review|'
        r'overview\s+of|comprehensive\s+(?:survey|review))\b',
        title_lower
    ))

    is_short = page_count > 0 and page_count < 4
    is_editorial = bool(re.search(r'\b(?:editorial|letter\s+to|workshop\s+report)\b', title_lower))

    return is_survey or is_short or is_editorial, is_survey, is_short


def screen_paper(key, title, pdf_path):
    """Screen a single paper. Returns dict with all criteria results."""
    text, pages = extract_text(pdf_path)

    if not text or len(text) < 200:
        return {
            'key': key, 'title': title, 'pages': pages,
            'decision': 'Uncertain', 'reason': 'text_extraction_failed',
            'ic1_codegen': 0, 'ic1_eff': 0, 'ic1_pass': False,
            'ic2_score': 0, 'ic2_pass': False,
            'ic3_score': 0, 'ic3_pass': False,
            'ic4_score': 0, 'ic4_pass': False,
            'ec1_flag': False, 'ec2_score': 0, 'ec2_flag': False,
            'ec3_flag': False, 'ec4_flag': False, 'ec4_survey': False, 'ec4_short': False,
        }

    ic1_cg, ic1_eff, ic1_pass = check_ic1_efficiency_for_codegen(text)
    ic2_score, ic2_pass = check_ic2_llm_based(text)
    ic3_score, ic3_pass = check_ic3_codegen_task(text)
    ic4_score, ic4_pass = check_ic4_efficiency_metric(text)

    ec1_flag = check_ec1_not_efficiency_focused(text, ic1_eff)
    ec2_score, ec2_flag = check_ec2_other_se_task(text)
    ec3_flag = check_ec3_not_english(text)
    ec4_flag, ec4_survey, ec4_short = check_ec4_survey_or_short(text, pages, title)

    # Decision logic
    # Exclude if any EC is triggered
    if ec3_flag:
        decision, reason = 'Exclude', 'EC3:not_english'
    elif ec4_flag:
        reasons = []
        if ec4_survey: reasons.append('survey')
        if ec4_short: reasons.append('short')
        decision, reason = 'Exclude', f'EC4:{"+".join(reasons)}'
    elif ec2_flag:
        decision, reason = 'Exclude', 'EC2:other_se_task'
    elif not ic2_pass:
        decision, reason = 'Exclude', 'IC2:not_llm_based'
    elif not ic3_pass and ic1_cg < 1:
        decision, reason = 'Exclude', 'IC3:not_codegen_task'
    elif not ic1_pass and ic1_eff < 2 and ic1_cg < 1:
        decision, reason = 'Exclude', 'IC1:not_efficiency_for_codegen'
    elif ec1_flag and ic4_score < 1:
        decision, reason = 'Exclude', 'EC1:efficiency_not_core'
    elif ic1_pass and ic2_pass and ic3_pass and ic4_pass:
        decision, reason = 'Include', 'all_IC_passed'
    elif (ic1_cg >= 1 and ic1_eff >= 2 and ic2_pass and
          (ic3_pass or ic3_score >= 1) and (ic4_pass or ic4_score >= 1)):
        decision, reason = 'Include', 'strong_match'
    elif ic1_pass and ic2_pass:
        decision, reason = 'Uncertain', 'partial_IC_match'
    else:
        decision, reason = 'Uncertain', 'weak_match'

    return {
        'key': key, 'title': title, 'pages': pages,
        'decision': decision, 'reason': reason,
        'ic1_codegen': ic1_cg, 'ic1_eff': ic1_eff, 'ic1_pass': ic1_pass,
        'ic2_score': ic2_score, 'ic2_pass': ic2_pass,
        'ic3_score': ic3_score, 'ic3_pass': ic3_pass,
        'ic4_score': ic4_score, 'ic4_pass': ic4_pass,
        'ec1_flag': ec1_flag, 'ec2_score': ec2_score, 'ec2_flag': ec2_flag,
        'ec3_flag': ec3_flag, 'ec4_flag': ec4_flag, 'ec4_survey': ec4_survey,
        'ec4_short': ec4_short,
    }


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"processed": []}


def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)


FIELDNAMES = [
    'key', 'title', 'pages', 'decision', 'reason',
    'ic1_codegen', 'ic1_eff', 'ic1_pass',
    'ic2_score', 'ic2_pass',
    'ic3_score', 'ic3_pass',
    'ic4_score', 'ic4_pass',
    'ec1_flag', 'ec2_score', 'ec2_flag',
    'ec3_flag', 'ec4_flag', 'ec4_survey', 'ec4_short',
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--offset', type=int, default=0, help='Start from this index')
    parser.add_argument('--limit', type=int, default=0, help='Process at most N papers')
    parser.add_argument('--batch-id', type=str, default='', help='Batch identifier for parallel runs')
    args = parser.parse_args()

    # Load T&A screening results and candidate metadata
    ta_labels = {}
    with open(TA_FILE, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            ta_labels[r['key']] = r['label']

    candidates = {}
    with open(CANDIDATES_FILE, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            candidates[r['key']] = r

    # Build list of papers to screen: T&A passed + have PDF
    to_screen = []
    for key, label in ta_labels.items():
        if label not in ('Likely-Include', 'Uncertain'):
            continue
        pdf_path = os.path.join(PDF_DIR, safe_filename(key))
        if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) < 1000:
            continue
        title = candidates.get(key, {}).get('title', ta_labels.get(key, ''))
        # Get title from TA file if not in candidates
        to_screen.append((key, title, pdf_path))

    # Also get titles from TA CSV
    ta_titles = {}
    with open(TA_FILE, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            ta_titles[r['key']] = r.get('title', '')

    for i, (key, title, pdf_path) in enumerate(to_screen):
        if not title and key in ta_titles:
            to_screen[i] = (key, ta_titles[key], pdf_path)

    # Apply offset and limit
    if args.offset > 0:
        to_screen = to_screen[args.offset:]
    if args.limit > 0:
        to_screen = to_screen[:args.limit]

    print(f"Papers to screen: {len(to_screen)}")

    # Load progress
    progress = load_progress()
    processed_keys = set(progress['processed'])

    # Output file
    output_file = OUTPUT_FILE
    if args.batch_id:
        output_file = os.path.join(SCRIPT_DIR, f"screening-ft-batch-{args.batch_id}.csv")

    # Check if output file exists (append mode)
    file_exists = os.path.exists(output_file)

    results = []
    stats = Counter()

    with open(output_file, 'a', newline='', encoding='utf-8') as outf:
        writer = csv.DictWriter(outf, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        for i, (key, title, pdf_path) in enumerate(to_screen):
            if key in processed_keys:
                continue

            result = screen_paper(key, title, pdf_path)
            writer.writerow(result)
            outf.flush()

            progress['processed'].append(key)
            stats[result['decision']] += 1
            results.append(result)

            if (i + 1) % 50 == 0:
                save_progress(progress)
                total_done = len(progress['processed'])
                print(f"[{i+1}/{len(to_screen)}] "
                      f"Include={stats['Include']} Exclude={stats['Exclude']} "
                      f"Uncertain={stats['Uncertain']} | total_processed={total_done}")

    save_progress(progress)

    # Final summary
    print(f"\n{'='*60}")
    print(f"SCREENING SUMMARY (batch={args.batch_id or 'main'})")
    print(f"{'='*60}")
    print(f"  Processed: {len(results)}")
    for d, c in stats.most_common():
        print(f"  {d}: {c}")

    # Reason breakdown
    reasons = Counter(r['reason'] for r in results)
    print(f"\nExclusion reasons:")
    for r, c in reasons.most_common(15):
        print(f"  {r}: {c}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
