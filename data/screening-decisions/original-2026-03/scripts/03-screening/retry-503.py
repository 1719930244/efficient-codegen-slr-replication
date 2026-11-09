#!/usr/bin/env python3
"""
Retry papers that failed with HTTP 503 during the first LLM screening round.
Reads existing batch CSVs to find 503 keys, then re-screens using the same logic.
"""

import csv
import json
import os
import re
import subprocess
import sys
import time

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..')
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
CANDIDATES_FILE = os.path.join(DATA_DIR, "candidates-planb.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "screening-ft-llm-retry.csv")
PROGRESS_FILE = os.path.join(DATA_DIR, "screening-ft-llm-retry-progress.json")
LOG_FILE = os.path.join(DATA_DIR, "screening-ft-llm-log.jsonl")

API_BASE = "https://code.mmkg.cloud"
API_KEY = "sk-f3c5dda1d52aa946dc923d5ab6adf3f13f0fb2b2efcf0e90cbae037ffae72814"
MODEL = "gpt-5.2-codex"

SYSTEM_PROMPT = """You are a systematic literature review screener. You evaluate academic papers against inclusion/exclusion criteria for a survey on "Efficiency in LLM-based Code Generation".

For each paper, you must give a binary YES/NO judgment on each criterion below, then a final INCLUDE/EXCLUDE/UNCERTAIN decision.

## Inclusion Criteria (ALL must be YES to include)

- IC1: The paper proposes or evaluates a technique that improves the EFFICIENCY of CODE GENERATION, covering the data, training, or inference stage. The paper's core contribution must be about making code generation more efficient (faster, cheaper, smaller, less memory, etc.), not just mentioning efficiency in passing.
- IC2: The paper uses an LLM-based or Transformer-based approach. Papers using only traditional ML, rule-based, or template-based methods are excluded.
- IC3: The paper addresses code completion or natural-language-to-code generation tasks. Papers about other tasks (image generation, text generation, code review, bug detection, test generation, code translation, etc.) that merely mention "code generation" are excluded.
- IC4: The paper reports at least one quantitative efficiency metric (latency, memory, throughput, FLOPs, energy, cost, training time, speedup, model size reduction, etc.) as part of its evaluation.

## Exclusion Criteria (ANY triggers exclusion)

- EC1: The core technique does NOT target efficiency. The paper may report inference time incidentally but efficiency is not a design objective or contribution.
- EC2: The paper targets SE tasks OTHER than code generation (e.g., defect detection, test generation, code review, program repair, code summarization).
- EC3: The paper is not written in English.
- EC4: The paper is a survey, editorial, tutorial, or short abstract (fewer than 4 pages).

## Output Format (strict JSON)

Return ONLY a JSON object, no other text:
{"IC1": "YES/NO", "IC2": "YES/NO", "IC3": "YES/NO", "IC4": "YES/NO", "EC1": "YES/NO", "EC2": "YES/NO", "EC3": "YES/NO", "EC4": "YES/NO", "decision": "INCLUDE/EXCLUDE/UNCERTAIN", "reason": "<brief one-line reason>"}

Rules:
- decision=INCLUDE only if IC1-IC4 are all YES and EC1-EC4 are all NO
- decision=EXCLUDE if any IC is NO or any EC is YES
- decision=UNCERTAIN only if the text is too short/corrupted to judge confidently"""


def safe_filename(key):
    return re.sub(r'[^\w\-.]', '_', key) + '.pdf'


def extract_text(pdf_path, max_pages=4):
    try:
        result = subprocess.run(
            ["pdftotext", "-l", str(max_pages), pdf_path, "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30
        )
        return result.stdout.decode('utf-8', errors='replace').strip()
    except Exception:
        return ""


def call_llm(title, text):
    if len(text) > 4000:
        text = text[:3000] + "\n...[truncated]...\n" + text[-800:]

    user_msg = "Title: {}\n\nPaper text (first pages):\n{}".format(title, text)

    payload = {
        "model": MODEL,
        "instructions": SYSTEM_PROMPT,
        "input": [{"role": "user", "content": user_msg}],
        "temperature": 0,
        "max_output_tokens": 300,
    }
    headers = {
        "Authorization": "Bearer " + API_KEY,
        "Content-Type": "application/json",
    }

    for attempt in range(3):
        try:
            resp = requests.post(
                API_BASE + "/v1/responses",
                json=payload,
                headers=headers,
                timeout=120,
            )
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print("  Rate limited, waiting {}s...".format(wait))
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                return None, "http_{}".format(resp.status_code)

            data = resp.json()
            content = ""
            for o in data.get("output", []):
                for c in o.get("content", []):
                    if c.get("type") == "output_text":
                        content = c.get("text", "").strip()
                        break
                if content:
                    break

            if not content:
                return None, "empty_response"

            if content.startswith("```"):
                content = re.sub(r'^```\w*\n?', '', content)
                content = re.sub(r'\n?```$', '', content)
            content = content.strip()

            result = json.loads(content)
            return result, None

        except (ValueError, KeyError) as e:
            return None, "parse_error: {}".format(str(e)[:50])
        except requests.exceptions.Timeout:
            if attempt < 2:
                time.sleep(5)
                continue
            return None, "timeout"
        except Exception as e:
            return None, "error: {}".format(str(e)[:50])

    return None, "max_retries"


FIELDNAMES = [
    'key', 'title', 'decision', 'reason',
    'IC1', 'IC2', 'IC3', 'IC4',
    'EC1', 'EC2', 'EC3', 'EC4',
    'error',
]


def log_record(key, title, text_snippet, llm_result, error, decision):
    import datetime
    record = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'key': key,
        'title': title,
        'input_text_length': len(text_snippet) if text_snippet else 0,
        'input_text_preview': (text_snippet[:500] + '...') if text_snippet and len(text_snippet) > 500 else text_snippet,
        'llm_raw_response': llm_result,
        'error': error,
        'final_decision': decision,
        'model': MODEL,
        'retry': True,
    }
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--delay', type=float, default=0.5)
    parser.add_argument('--batch-id', type=str, default='')
    args = parser.parse_args()

    # Collect 503 keys from existing batch CSVs
    keys_503 = []
    for b in ['b1', 'b2', 'b3', 'b4']:
        batch_file = os.path.join(DATA_DIR, "screening-ft-llm-batch-{}.csv".format(b))
        if os.path.exists(batch_file):
            with open(batch_file, 'r', encoding='utf-8') as f:
                for r in csv.DictReader(f):
                    if r.get('error', '').strip() == 'http_503':
                        keys_503.append(r['key'])

    print("Total 503 papers to retry: {}".format(len(keys_503)))

    # Load candidate metadata
    candidates = {}
    with open(CANDIDATES_FILE, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            candidates[r['key']] = r

    # Build retry list with PDF paths
    to_screen = []
    for key in keys_503:
        pdf_path = os.path.join(PDF_DIR, safe_filename(key))
        if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) < 1000:
            continue
        title = candidates.get(key, {}).get('title', key)
        to_screen.append((key, title, pdf_path))

    # Apply offset/limit
    if args.offset > 0:
        to_screen = to_screen[args.offset:]
    if args.limit > 0:
        to_screen = to_screen[:args.limit]

    print("Papers with valid PDF: {}".format(len(to_screen)))

    # Load progress
    output_file = OUTPUT_FILE
    progress_file = PROGRESS_FILE
    if args.batch_id:
        output_file = os.path.join(DATA_DIR, "screening-ft-llm-retry-{}.csv".format(args.batch_id))
        progress_file = os.path.join(DATA_DIR, "screening-ft-llm-retry-progress-{}.json".format(args.batch_id))

    processed = {}
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            progress = json.load(f)
            processed = progress.get('processed', {})
    else:
        progress = {"processed": {}}

    file_exists = os.path.exists(output_file)
    stats = {"INCLUDE": 0, "EXCLUDE": 0, "UNCERTAIN": 0, "ERROR": 0}

    with open(output_file, 'a', newline='', encoding='utf-8') as outf:
        writer = csv.DictWriter(outf, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        for i, (key, title, pdf_path) in enumerate(to_screen):
            if key in processed:
                continue

            text = extract_text(pdf_path)
            if not text or len(text) < 200:
                row = {
                    'key': key, 'title': title,
                    'decision': 'UNCERTAIN', 'reason': 'text_extraction_failed',
                    'IC1': '', 'IC2': '', 'IC3': '', 'IC4': '',
                    'EC1': '', 'EC2': '', 'EC3': '', 'EC4': '',
                    'error': 'no_text',
                }
                writer.writerow(row)
                outf.flush()
                log_record(key, title, text, None, 'no_text', 'UNCERTAIN')
                processed[key] = 'no_text'
                stats['UNCERTAIN'] += 1
            else:
                result, error = call_llm(title, text)
                if result:
                    decision = result.get('decision', 'UNCERTAIN').upper()
                    row = {
                        'key': key, 'title': title,
                        'decision': decision,
                        'reason': result.get('reason', ''),
                        'IC1': result.get('IC1', ''),
                        'IC2': result.get('IC2', ''),
                        'IC3': result.get('IC3', ''),
                        'IC4': result.get('IC4', ''),
                        'EC1': result.get('EC1', ''),
                        'EC2': result.get('EC2', ''),
                        'EC3': result.get('EC3', ''),
                        'EC4': result.get('EC4', ''),
                        'error': '',
                    }
                    stats[decision] = stats.get(decision, 0) + 1
                else:
                    decision = 'UNCERTAIN'
                    row = {
                        'key': key, 'title': title,
                        'decision': 'UNCERTAIN', 'reason': error,
                        'IC1': '', 'IC2': '', 'IC3': '', 'IC4': '',
                        'EC1': '', 'EC2': '', 'EC3': '', 'EC4': '',
                        'error': error,
                    }
                    stats['ERROR'] += 1

                log_record(key, title, text, result, error, decision)
                writer.writerow(row)
                outf.flush()
                processed[key] = decision if result else 'error'

            progress['processed'] = processed

            # Save progress every 20
            if (i + 1) % 20 == 0:
                tmp = progress_file + ".tmp"
                with open(tmp, 'w') as pf:
                    json.dump(progress, pf)
                os.rename(tmp, progress_file)
                total = sum(stats.values())
                print("[{}/{}] INC={} EXC={} UNC={} ERR={} | total={}".format(
                    i + 1, len(to_screen),
                    stats['INCLUDE'], stats['EXCLUDE'],
                    stats['UNCERTAIN'], stats['ERROR'], total))

            time.sleep(args.delay)

    # Final save
    tmp = progress_file + ".tmp"
    with open(tmp, 'w') as pf:
        json.dump(progress, pf)
    os.rename(tmp, progress_file)

    print("\n" + "=" * 60)
    print("RETRY SCREENING SUMMARY (batch={})".format(args.batch_id or 'main'))
    print("Model: {}".format(MODEL))
    print("=" * 60)
    for k, v in sorted(stats.items()):
        print("  {}: {}".format(k, v))
    total = sum(stats.values())
    print("  Total: {}".format(total))
    if stats['INCLUDE'] > 0:
        print("  Include rate: {:.1f}%".format(stats['INCLUDE'] / max(total, 1) * 100))
    print("=" * 60)


if __name__ == '__main__':
    main()
