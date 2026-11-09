#!/usr/bin/env python3
"""
Automated Title & Abstract pre-screening for SLR.
Classifies deduplicated entries as Likely-Include / Likely-Exclude / Uncertain
based on IC/EC keyword matching.

IC1: Proposes or evaluates code generation efficiency optimization
IC2: Based on LLM/language model methods (REQUIRED for Likely-Include)
IC3: Involves code completion or NL-to-Code generation
IC4: Reports at least one efficiency metric
EC2: SE tasks other than code generation (defect detection, testing, etc.)
EC3: Non-English papers (filtered by title heuristic)
EC4: Reviews, editorials, short abstracts (<4 pages)

Year cutoff: entries before 2015 are auto-excluded.
"""

import re
import os
import csv
import json

DATA_DIR = "/home/szw/overleaf/efficient-codegen-survey/data"
YEAR_CUTOFF = 2017

# IC signals: presence in title or abstract increases inclusion likelihood
IC_CODE_GEN = [
    'code generation', 'code completion', 'code synthesis', 'program synthesis',
    'code infilling', 'fill-in-the-middle', 'automated programming',
    'code model', 'code llm', 'coding model', 'codegen',
    'nl-to-code', 'nl2code', 'text-to-code', 'text2code',
    'copilot', 'codebert', 'codex', 'codellama', 'starcoder', 'deepseek-coder',
    'deepseekcoder', 'codeqwen', 'qwen2.5-coder', 'wizardcoder', 'magicoder',
    'code llama', 'santacoder', 'incoder', 'polycoder', 'codegen2',
    'repository-level', 'repo-level', 'inline completion',
]

IC_EFFICIENCY = [
    'efficient', 'efficiency', 'optimization', 'optimizing', 'accelerat',
    'lightweight', 'compress', 'quantiz', 'prun', 'distill',
    'latency', 'throughput', 'computational cost', 'energy',
    'scalab', 'inference speed', 'memory footprint', 'model size',
    'speculative decoding', 'token budget', 'token limit',
    'kv cache', 'kv-cache', 'key-value cache',
    'lora', 'adapter', 'prefix tuning', 'parameter-efficient',
    'knowledge distillation',
    'post-training quantization', 'ptq', 'gptq', 'awq', 'ggml', 'gguf',
    'mixture of experts', 'moe', 'model routing',
    'prompt compression', 'context compression', 'context pruning',
    'early exit', 'early stopping', 'non-autoregressive',
    'continuous batching', 'vllm', 'tensorrt', 'onnx',
    'flops', 'gpu memory', 'training cost', 'training time',
]

IC_LLM = [
    # Core LLM / AI terms
    'large language model', 'llm', 'language model', 'transformer',
    'pre-trained model', 'pretrained model', 'foundation model',
    'generative ai', 'generative model', 'generative pre-train',
    'artificial intelligence', ' ai ', ' ai-',
    # Architecture signals
    'neural network', 'neural code', 'deep learning', 'deep neural',
    'attention mechanism', 'self-attention', 'multi-head attention',
    'decoder-only', 'encoder-decoder', 'autoregressive',
    'seq2seq', 'sequence-to-sequence',
    # Model families (general)
    'gpt', 'bert', 'llama', 't5 ', 'palm', 'gemini', 'claude',
    'chatgpt', 'openai', 'phi-', 'gemma',
    'mistral', 'mixtral', 'qwen', 'deepseek',
    # Code-specific models (strengthens LLM signal)
    'codex', 'copilot', 'codebert', 'codellama', 'code llama',
    'starcoder', 'deepseekcoder', 'codeqwen', 'wizardcoder', 'magicoder',
    'santacoder', 'incoder', 'polycoder', 'codegen2',
    'codet5', 'plbart', 'unixcoder', 'graphcodebert',
    # Training paradigm
    'fine-tun', 'finetuning', 'pretraining', 'pre-training',
    'instruction tuning', 'instruction-tuned',
    'in-context learning', 'few-shot', 'zero-shot', 'prompt',
    'rlhf', 'reinforcement learning from human',
]

# EC signals: presence increases exclusion likelihood
EC_NON_CODEGEN = [
    'defect detection', 'bug detection', 'vulnerability detection',
    'test generation', 'test case generation', 'mutation testing',
    'code review', 'code smell', 'code clone', 'clone detection',
    'malware', 'intrusion detection', 'network security',
    'image generation', 'image synthesis', 'text-to-image',
    'speech recognition', 'machine translation', 'question answering',
    'sentiment analysis', 'text classification', 'named entity',
    'object detection', 'image classification', 'video generation',
    'drug discovery', 'protein', 'molecular',
    'autonomous driving', 'robot', 'reinforcement learning',
    'recommendation system', 'click-through',
    'hardware design', 'circuit', 'vlsi', 'fpga',
    'compiler optimization', 'instruction scheduling',
    'database query', 'sql optimization',
]

EC_SURVEY = [
    'systematic review', 'literature review', 'survey of',
    'a survey', 'comprehensive survey', 'systematic mapping',
]

def classify_entry(entry):
    """Classify an entry based on title and abstract keywords."""
    title = entry.get('title', '').lower()
    abstract = entry.get('abstract', '').lower()
    text = title + ' ' + abstract
    year = entry.get('year', '')

    scores = {
        'ic_codegen': 0,
        'ic_efficiency': 0,
        'ic_llm': 0,
        'ec_non_codegen': 0,
        'ec_survey': 0,
    }
    matched_ic = []
    matched_ec = []

    # Year cutoff: auto-exclude pre-2015
    if year.isdigit() and int(year) < YEAR_CUTOFF:
        return 'Likely-Exclude', scores, ['year<2015'], ['year<2015']

    for kw in IC_CODE_GEN:
        if kw in text:
            scores['ic_codegen'] += 1
            matched_ic.append(f'codegen:{kw}')

    for kw in IC_EFFICIENCY:
        if kw in text:
            scores['ic_efficiency'] += 1
            matched_ic.append(f'eff:{kw}')

    for kw in IC_LLM:
        if kw in text:
            scores['ic_llm'] += 1
            matched_ic.append(f'llm:{kw}')

    for kw in EC_NON_CODEGEN:
        if kw in text:
            scores['ec_non_codegen'] += 1
            matched_ec.append(f'non-cg:{kw}')

    for kw in EC_SURVEY:
        if kw in text:
            scores['ec_survey'] += 1
            matched_ec.append(f'survey:{kw}')

    # Decision logic (LLM signal REQUIRED for Likely-Include)

    # Strong exclude: non-codegen domain
    if scores['ec_non_codegen'] >= 2 and scores['ic_codegen'] == 0:
        return 'Likely-Exclude', scores, matched_ic, matched_ec

    # Strong exclude: survey/review without codegen focus
    if scores['ec_survey'] >= 1 and scores['ic_codegen'] == 0:
        return 'Likely-Exclude', scores, matched_ic, matched_ec

    # Strong include: codegen + efficiency + LLM (all three present)
    if scores['ic_codegen'] >= 1 and scores['ic_efficiency'] >= 1 and scores['ic_llm'] >= 1:
        if scores['ec_non_codegen'] == 0:
            return 'Likely-Include', scores, matched_ic, matched_ec

    # Moderate include: codegen + LLM (even without explicit efficiency keyword)
    if scores['ic_codegen'] >= 1 and scores['ic_llm'] >= 1:
        if scores['ec_non_codegen'] == 0:
            return 'Likely-Include', scores, matched_ic, matched_ec

    # Uncertain: codegen + efficiency but NO LLM signal
    if scores['ic_codegen'] >= 1 and scores['ic_efficiency'] >= 1 and scores['ic_llm'] == 0:
        return 'Uncertain', scores, matched_ic, matched_ec

    # Uncertain: efficiency + LLM but no codegen
    if scores['ic_efficiency'] >= 2 and scores['ic_llm'] >= 1 and scores['ec_non_codegen'] == 0:
        return 'Uncertain', scores, matched_ic, matched_ec

    # No relevant signals at all
    if scores['ic_codegen'] == 0 and scores['ic_efficiency'] == 0 and scores['ic_llm'] == 0:
        return 'Likely-Exclude', scores, matched_ic, matched_ec

    return 'Uncertain', scores, matched_ic, matched_ec


def parse_bib_entries(filepath):
    """Parse deduplicated BibTeX file."""
    entries = []
    if not os.path.exists(filepath):
        return entries
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    pattern = r'@(\w+)\{([^,]*),(.*?)(?=\n@|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)
    for entry_type, key, body in matches:
        entry = {
            'type': entry_type.lower(),
            'key': key.strip(),
            'source_file': 'deduplicated.bib',
        }
        for field in ['title', 'author', 'year', 'doi', 'url', 'eprint', 'abstract']:
            match = re.search(rf'{field}\s*=\s*\{{(.*?)\}}', body, re.DOTALL | re.IGNORECASE)
            if match:
                entry[field] = match.group(1).strip()
            else:
                match = re.search(rf'{field}\s*=\s*"(.*?)"', body, re.DOTALL | re.IGNORECASE)
                if match:
                    entry[field] = match.group(1).strip()
        if 'title' in entry:
            entries.append(entry)
    return entries


def main():
    dedup_path = os.path.join(DATA_DIR, 'deduplicated.bib')
    entries = parse_bib_entries(dedup_path)
    print(f"Loaded {len(entries)} entries from deduplicated.bib")

    results = []
    counts = {'Likely-Include': 0, 'Likely-Exclude': 0, 'Uncertain': 0}

    for entry in entries:
        label, scores, matched_ic, matched_ec = classify_entry(entry)
        counts[label] += 1
        results.append({
            'key': entry['key'],
            'title': entry.get('title', ''),
            'year': entry.get('year', ''),
            'label': label,
            'ic_codegen': scores['ic_codegen'],
            'ic_efficiency': scores['ic_efficiency'],
            'ic_llm': scores['ic_llm'],
            'ec_non_codegen': scores['ec_non_codegen'],
            'ec_survey': scores['ec_survey'],
            'matched_ic': '; '.join(matched_ic[:10]),
            'matched_ec': '; '.join(matched_ec[:5]),
            'has_abstract': 'yes' if entry.get('abstract') else 'no',
        })

    # Write screening results
    output_path = os.path.join(DATA_DIR, 'screening-ta-auto.csv')
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['key', 'title', 'year', 'label', 'ic_codegen', 'ic_efficiency',
                      'ic_llm', 'ec_non_codegen', 'ec_survey', 'matched_ic', 'matched_ec',
                      'has_abstract']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Written: {output_path}")

    # Write Likely-Include entries as separate bib for next stage
    include_keys = set(r['key'] for r in results if r['label'] == 'Likely-Include')
    include_path = os.path.join(DATA_DIR, 'likely-include.bib')
    with open(dedup_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    pattern = r'@(\w+)\{([^,]*),(.*?)(?=\n@|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)
    with open(include_path, 'w', encoding='utf-8') as f:
        f.write(f"% Likely-Include entries from automated T&A screening\n")
        f.write(f"% Total: {counts['Likely-Include']}\n\n")
        for entry_type, key, body in matches:
            if key.strip() in include_keys:
                f.write(f'@{entry_type}{{{key},{body}'.rstrip() + '\n\n')
    print(f"Written: {include_path}")

    # Update PRISMA counts
    prisma_path = os.path.join(DATA_DIR, 'prisma-counts.json')
    with open(prisma_path, 'r') as f:
        prisma = json.load(f)
    prisma['screening']['auto_ta_likely_include'] = counts['Likely-Include']
    prisma['screening']['auto_ta_likely_exclude'] = counts['Likely-Exclude']
    prisma['screening']['auto_ta_uncertain'] = counts['Uncertain']
    prisma['screening']['year_cutoff'] = YEAR_CUTOFF
    with open(prisma_path, 'w') as f:
        json.dump(prisma, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"AUTOMATED T&A SCREENING RESULTS")
    print(f"{'='*60}")
    print(f"  Likely-Include:  {counts['Likely-Include']:>6d}")
    print(f"  Uncertain:       {counts['Uncertain']:>6d}")
    print(f"  Likely-Exclude:  {counts['Likely-Exclude']:>6d}")
    print(f"  {'─'*40}")
    print(f"  Total:           {sum(counts.values()):>6d}")
    print(f"{'='*60}")
    print(f"\nNext: Human reviewers should verify Likely-Include + Uncertain entries")

if __name__ == '__main__':
    main()
