#!/usr/bin/env python3
"""
Fetch missing abstracts for Plan B candidates.
Strategy:
  1. arXiv API (for papers with arxiv_id)
  2. Semantic Scholar API (by DOI, arXiv ID, or title search)
Saves results to abstracts-supplemented.json (resumable).
"""

import csv
import json
import os
import re
import time
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET

DATA_DIR = "/home/szw/overleaf/efficient-codegen-survey/data"
CANDIDATES = os.path.join(DATA_DIR, "candidates-planb.csv")
BIB_FILE = os.path.join(DATA_DIR, "deduplicated.bib")
OUTPUT = os.path.join(DATA_DIR, "abstracts-supplemented.json")

USER_AGENT = "SLR-Abstract-Fetcher/1.0 (academic survey)"
S2_DELAY = 0.2
ARXIV_DELAY = 0.5


def load_progress():
    if os.path.exists(OUTPUT):
        with open(OUTPUT) as f:
            return json.load(f)
    return {}


def save_progress(data):
    with open(OUTPUT, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_bib_abstracts():
    """Extract existing abstracts from bib file."""
    abstracts = {}
    with open(BIB_FILE, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    pattern = r'@(\w+)\{([^,]*),(.*?)(?=\n@|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)
    for entry_type, key, body in matches:
        k = key.strip()
        ab_match = re.search(r'abstract\s*=\s*\{(.*?)\}', body, re.DOTALL | re.IGNORECASE)
        if ab_match:
            ab = ab_match.group(1).strip()
            if len(ab) > 50:
                abstracts[k] = ab
    return abstracts


def fetch_arxiv_abstract(arxiv_id):
    """Fetch abstract from arXiv API."""
    aid = arxiv_id.strip()
    if aid.startswith('http'):
        m = re.search(r'(\d{4}\.\d{4,5})(v\d+)?', aid)
        if m:
            aid = m.group(1)
    url = f"http://export.arxiv.org/api/query?id_list={aid}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode('utf-8')
        # Parse XML
        root = ET.fromstring(data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', ns)
        if entries:
            summary = entries[0].find('atom:summary', ns)
            if summary is not None and summary.text:
                ab = summary.text.strip()
                ab = re.sub(r'\s+', ' ', ab)
                if len(ab) > 50:
                    return ab
    except Exception:
        pass
    return None


def fetch_s2_abstract(identifier, by_title=False):
    """Fetch abstract from Semantic Scholar API."""
    if by_title:
        query = urllib.parse.quote(identifier[:200])
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=1&fields=abstract"
    else:
        url = f"https://api.semanticscholar.org/graph/v1/paper/{identifier}?fields=abstract"

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        if by_title:
            papers = data.get('data', [])
            if papers:
                data = papers[0]
            else:
                return None
        ab = data.get('abstract', '')
        if ab and len(ab) > 50:
            return ab
    except urllib.error.HTTPError as e:
        if e.code == 429:
            time.sleep(2)
        return None
    except Exception:
        pass
    return None


def main():
    # Load candidates
    entries = []
    with open(CANDIDATES, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            entries.append(r)
    print(f"Total candidates: {len(entries)}")

    # Load existing bib abstracts
    bib_abstracts = parse_bib_abstracts()
    print(f"Abstracts already in bib: {len(bib_abstracts)}")

    # Load progress
    progress = load_progress()
    print(f"Previously fetched: {len(progress)}")

    # Find entries needing abstracts
    to_fetch = []
    already_have = 0
    for e in entries:
        key = e['key']
        if key in bib_abstracts or key in progress:
            already_have += 1
            continue
        to_fetch.append(e)

    print(f"Already have abstract: {already_have}")
    print(f"Need to fetch: {len(to_fetch)}")

    if not to_fetch:
        print("All abstracts available!")
        return

    success = 0
    fail = 0
    for i, entry in enumerate(to_fetch):
        key = entry['key']
        arxiv_id = entry.get('arxiv_id', '').strip()
        doi = entry.get('doi', '').strip()
        title = entry.get('title', '').strip()
        abstract = None

        # Strategy 1: arXiv API
        if arxiv_id and not abstract:
            abstract = fetch_arxiv_abstract(arxiv_id)
            if abstract:
                progress[key] = {"abstract": abstract, "source": "arxiv_api"}
                success += 1
            time.sleep(ARXIV_DELAY)

        # Strategy 2: S2 by DOI
        if doi and not abstract:
            abstract = fetch_s2_abstract(f"DOI:{doi}")
            if abstract:
                progress[key] = {"abstract": abstract, "source": "s2_doi"}
                success += 1
            time.sleep(S2_DELAY)

        # Strategy 3: S2 by arXiv ID
        if arxiv_id and not abstract:
            aid = arxiv_id
            if not aid.startswith('ARXIV:'):
                aid = f"ARXIV:{aid}"
            abstract = fetch_s2_abstract(aid)
            if abstract:
                progress[key] = {"abstract": abstract, "source": "s2_arxiv"}
                success += 1
            time.sleep(S2_DELAY)

        # Strategy 4: S2 title search
        if title and not abstract:
            abstract = fetch_s2_abstract(title, by_title=True)
            if abstract:
                progress[key] = {"abstract": abstract, "source": "s2_title"}
                success += 1
            time.sleep(S2_DELAY)

        if not abstract:
            progress[key] = {"abstract": "", "source": "not_found"}
            fail += 1

        # Progress report every 100
        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(to_fetch)}] success={success} fail={fail}")
            save_progress(progress)

    save_progress(progress)

    # Summary
    sources = {}
    for v in progress.values():
        src = v.get('source', 'unknown')
        sources[src] = sources.get(src, 0) + 1

    print(f"\n{'='*60}")
    print(f"ABSTRACT FETCH SUMMARY")
    print(f"{'='*60}")
    print(f"  From bib:      {len(bib_abstracts)}")
    for src, cnt in sorted(sources.items()):
        print(f"  {src:15s}: {cnt}")
    total_have = len(bib_abstracts) + sum(1 for v in progress.values() if v.get('abstract'))
    print(f"  Total with abstract: {total_have}/{len(entries)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
