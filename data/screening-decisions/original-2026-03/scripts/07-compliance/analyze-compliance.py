#!/usr/bin/env python3
"""
Analyze reporting compliance of 125 primary studies.
Extracts text from each PDF and searches for metric-related keywords.
"""

import csv
import os
import re
import json
import sys

try:
    import PyPDF2
except ImportError:
    print("PyPDF2 not found, trying pymupdf...")
    import fitz

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..')
PDF_DIR = os.path.join(DATA_DIR, 'pdfs')
CSV_PATH = os.path.join(DATA_DIR, 'fine-grained-classification.csv')

# Key name resolution: CSV key -> PDF filename
def resolve_pdf_path(key):
    """Try to find the PDF file for a given paper key."""
    candidates = [
        os.path.join(PDF_DIR, f'{key}.pdf'),
    ]
    # Strip prefixes
    stripped = key
    for prefix in ['arxiv_', 'doi_']:
        if key.startswith(prefix):
            stripped = key[len(prefix):]
            break
    # For doi_ keys, also try stripping deeper
    if key.startswith('doi_'):
        # e.g. doi_10_48550_arXiv_2402_07844 -> extract arxiv id
        m = re.search(r'arXiv[_/](\d{4}[_\.]\d{4,5})', key)
        if m:
            arxiv_id = m.group(1).replace('.', '_')
            candidates.append(os.path.join(PDF_DIR, f'{arxiv_id}.pdf'))
    candidates.append(os.path.join(PDF_DIR, f'{stripped}.pdf'))

    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def extract_text_pypdf2(pdf_path):
    """Extract text from PDF using PyPDF2."""
    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception as e:
        print(f"  Error reading {pdf_path}: {e}", file=sys.stderr)
    return text.lower()


# ============ Metric detection patterns ============

def check_functional_correctness(text):
    """Check if paper reports functional correctness metrics."""
    patterns = [
        r'pass@\s*[k1-9]', r'pass\s*@\s*[k1-9]',
        r'humaneval', r'human[\s-]?eval',
        r'mbpp',
        r'code[\s-]?contest',
        r'apps\s+benchmark', r'apps\s+dataset',
        r'swe[\s-]?bench',
        r'functional\s+correct',
        r'test\s+case.*pass', r'pass.*test\s+case',
        r'exact\s+match', r'bleu', r'codebleu',
        r'execution[\s-]?based\s+evaluat',
    ]
    for p in patterns:
        if re.search(p, text):
            return True
    return False


def check_model_info(text):
    """Check if paper reports model name/version."""
    patterns = [
        r'codellama', r'code\s*llama',
        r'starcoder', r'star\s*coder',
        r'deepseek[\s-]?coder',
        r'codegen\b',
        r'gpt[\s-]?[34]', r'gpt[\s-]?3\.5', r'gpt[\s-]?4',
        r'claude',
        r'llama[\s-]?\d', r'llama\s+\d',
        r'gemini', r'gemma',
        r'phi[\s-]?\d',
        r'mistral', r'mixtral',
        r'qwen',
        r'codex\b',
        r'incoder',
        r'santacoder',
        r'wizardcoder',
        r'magicoder',
        r'codeqwen',
        r'yi[\s-]?coder',
        r'\d+[bm]\s+param', r'\d+[bm]\s+model',
        r'billion\s+param', r'million\s+param',
    ]
    count = 0
    for p in patterns:
        if re.search(p, text):
            count += 1
    return count >= 1


def check_hardware(text):
    """Check if paper reports hardware specification."""
    patterns = [
        r'a100', r'v100', r'h100', r'a6000', r'a40\b',
        r'rtx\s*\d{4}', r'rtx\s*[34]090',
        r'nvidia', r'geforce',
        r'tpu[\s-]?v\d',
        r'gpu\s+(memory|ram)', r'(\d+)\s*gb\s+(gpu|vram)',
        r'cuda\s+core',
        r'intel\s+xeon', r'amd\s+epyc',
        r'hardware\s+configur', r'hardware\s+setup',
        r'computation.*environment', r'experimental.*setup.*gpu',
    ]
    for p in patterns:
        if re.search(p, text):
            return True
    return False


def check_latency(text):
    """Check if paper reports latency/throughput metrics."""
    patterns = [
        r'latency', r'throughput',
        r'ttft\b', r'time[\s-]?to[\s-]?first[\s-]?token',
        r'tokens?\s*(per|/)\s*second', r'tok/s', r'tokens/s',
        r'inference\s+time', r'generation\s+time', r'decoding\s+time',
        r'wall[\s-]?clock\s+time',
        r'speedup\s*(ratio|factor)?[\s:]*[\d\.]', r'\d+[\.\d]*\s*[x×]\s*(speed|fast)',
        r'requests?\s*(per|/)\s*second',
        r'time\s+per\s+token',
        r'end[\s-]?to[\s-]?end\s+time',
        r'prefill\s+time', r'decode\s+time',
    ]
    for p in patterns:
        if re.search(p, text):
            return True
    return False


def check_memory(text):
    """Check if paper reports memory consumption."""
    patterns = [
        r'memory\s+(consumption|usage|footprint|overhead|requirement)',
        r'peak\s+memory', r'gpu\s+memory',
        r'vram\s+(usage|consumption)',
        r'(\d+)\s*(gb|mb)\s*(memory|ram|vram)',
        r'out[\s-]?of[\s-]?memory', r'oom\b',
        r'memory[\s-]?efficient',
        r'kv[\s-]?cache\s+size',
        r'model\s+size.*gb',
        r'memory\s+reduction',
    ]
    # Need to distinguish between "memory" as generic mention vs actual measurement
    # Require at least one measurement-related pattern
    measurement_patterns = [
        r'memory\s+(consumption|usage|footprint)',
        r'peak\s+memory',
        r'gpu\s+memory\s+usage',
        r'(\d+\.?\d*)\s*(gb|mb)\s*(of\s+)?(gpu\s+)?memory',
        r'memory.*(\d+\.?\d*)\s*(gb|mb)',
        r'memory\s+reduction.*\d+',
        r'memory\s+overhead.*\d+',
        r'kv[\s-]?cache\s+(memory|size).*\d+',
    ]
    for p in measurement_patterns:
        if re.search(p, text):
            return True
    return False


def check_serving_config(text):
    """Check if paper reports serving/deployment configuration."""
    patterns = [
        r'vllm', r'tgi\b', r'text[\s-]?generation[\s-]?inference',
        r'tensorrt[\s-]?llm', r'triton\s+inference',
        r'ollama', r'llama\.cpp', r'ggml', r'gguf',
        r'onnx\s+runtime',
        r'batch\s+size[\s:=]*\d+',
        r'max[\s_]?batch',
        r'tensor\s+parallel', r'pipeline\s+parallel',
        r'data\s+parallel.*\d+\s+gpu',
        r'serving\s+(framework|engine|system|config)',
        r'deployment\s+config',
        r'quantiz.*\d[\s-]?bit',  # quantization bit width as serving config
    ]
    for p in patterns:
        if re.search(p, text):
            return True
    return False


def check_monetary_cost(text):
    """Check if paper reports monetary cost."""
    patterns = [
        r'monetary\s+cost', r'api\s+cost',
        r'\$\s*[\d,]+\.?\d*',  # dollar amounts
        r'cost.*\$', r'\$.*cost',
        r'dollar', r'usd\b',
        r'pricing', r'price\s+per',
        r'cost[\s-]?per[\s-]?(token|query|request|call)',
        r'inference\s+cost.*\d',
        r'training\s+cost.*\d',
        r'gpu[\s-]?hour', r'compute[\s-]?hour',
        r'flop.*cost', r'cost.*flop',
    ]
    # Filter out false positives: "$k$" in pass@k contexts
    money_found = False
    for p in patterns:
        matches = re.finditer(p, text)
        for m in matches:
            context = text[max(0, m.start()-50):m.end()+50]
            # Skip pass@$k$ false positives
            if re.search(r'pass\s*@\s*\$?\s*k', context):
                continue
            money_found = True
            break
        if money_found:
            break
    return money_found


def check_energy(text):
    """Check if paper reports energy/carbon metrics."""
    patterns = [
        r'energy\s+(consumption|usage|cost|footprint|efficien)',
        r'power\s+(consumption|usage|draw)',
        r'carbon\s+(footprint|emission)',
        r'co2\b', r'co₂',
        r'kwh\b', r'kilowatt[\s-]?hour',
        r'watt[\s-]?hour',
        r'joule', r'\bwatt\b.*\d',
        r'green\s+ai', r'sustainable\s+ai',
        r'codecarbon', r'carbontracker',
        r'electricity\s+cost',
    ]
    for p in patterns:
        if re.search(p, text):
            return True
    return False


def analyze_paper(key, pdf_path):
    """Analyze a single paper for reporting compliance."""
    text = extract_text_pypdf2(pdf_path)
    if not text or len(text) < 100:
        return None  # Could not extract meaningful text

    return {
        'key': key,
        'functional_correctness': check_functional_correctness(text),
        'model_info': check_model_info(text),
        'hardware': check_hardware(text),
        'latency_throughput': check_latency(text),
        'memory': check_memory(text),
        'serving_config': check_serving_config(text),
        'monetary_cost': check_monetary_cost(text),
        'energy_carbon': check_energy(text),
        'text_length': len(text),
    }


def main():
    # Read paper list
    papers = []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            papers.append(row)

    print(f"Total papers: {len(papers)}")

    results = []
    missing = []
    errors = []

    for i, paper in enumerate(papers):
        key = paper['key']
        pdf_path = resolve_pdf_path(key)

        if not pdf_path:
            missing.append(key)
            continue

        result = analyze_paper(key, pdf_path)
        if result is None:
            errors.append(key)
            continue

        result['title'] = paper.get('title', '')
        result['rq'] = paper.get('new_rq', '')
        results.append(result)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(papers)}...", file=sys.stderr)

    # Compute statistics
    n = len(results)
    print(f"\nAnalyzed: {n} papers (missing PDFs: {len(missing)}, extraction errors: {len(errors)})")
    if missing:
        print(f"Missing: {', '.join(missing)}")
    if errors:
        print(f"Errors: {', '.join(errors)}")

    metrics = {
        'Functional correctness (pass@k)': 'functional_correctness',
        'Model name / version': 'model_info',
        'Hardware specification': 'hardware',
        'Latency / throughput': 'latency_throughput',
        'Memory consumption': 'memory',
        'Serving configuration': 'serving_config',
        'Monetary cost': 'monetary_cost',
        'Energy / carbon': 'energy_carbon',
    }

    print(f"\n{'Checklist Item':<40} {'Count':>6} {'Rate':>8}")
    print("-" * 58)
    for label, field in metrics.items():
        count = sum(1 for r in results if r[field])
        rate = count / n * 100
        print(f"{label:<40} {count:>6} {rate:>7.1f}%")

    # Breakdown by RQ
    print(f"\n\n=== Breakdown by RQ ===")
    rqs = sorted(set(r['rq'] for r in results))
    for rq in rqs:
        rq_results = [r for r in results if r['rq'] == rq]
        n_rq = len(rq_results)
        print(f"\n{rq} (n={n_rq}):")
        for label, field in metrics.items():
            count = sum(1 for r in rq_results if r[field])
            rate = count / n_rq * 100 if n_rq > 0 else 0
            print(f"  {label:<38} {count:>4} ({rate:>5.1f}%)")

    # Save detailed results
    output_path = os.path.join(os.path.dirname(__file__), 'compliance-results.json')
    with open(output_path, 'w') as f:
        json.dump({
            'summary': {label: sum(1 for r in results if r[field])
                       for label, field in metrics.items()},
            'total_analyzed': n,
            'missing_pdfs': missing,
            'errors': errors,
            'details': results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to {output_path}")

    # List papers that report monetary cost (for verification)
    print(f"\n\n=== Papers reporting monetary cost ===")
    for r in results:
        if r['monetary_cost']:
            print(f"  {r['key']}: {r['title'][:60]}")

    print(f"\n=== Papers reporting energy/carbon ===")
    for r in results:
        if r['energy_carbon']:
            print(f"  {r['key']}: {r['title'][:60]}")


if __name__ == '__main__':
    main()
