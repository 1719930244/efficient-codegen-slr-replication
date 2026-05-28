"""Monthly update search via Semantic Scholar Bulk API.

Replicates the structured search from sec-methodology.tex for the 2026-04 to
2026-05 window. arXiv API is too rate-limited for long Boolean queries;
Semantic Scholar Bulk endpoint indexes arXiv/IEEE/ACM/DBLP records together
and tolerates large queries.

IEEE/ACM/Scopus/Web of Science still require institutional credentials and
are deferred to the annual full sweep. This monthly procedure is limited to
the openly indexable subset surfaced by Semantic Scholar.
"""

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

PROXY = os.environ.get("HTTPS_PROXY") or "http://127.0.0.1:7890"
proxy_handler = urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})
opener = urllib.request.build_opener(proxy_handler)
opener.addheaders = [("User-Agent", "efficient-codegen-slr-monthly-update/1.0")]
urllib.request.install_opener(opener)

OUT_DIR = Path(__file__).parent.parent / "data" / "monthly-update-2026-05"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_START = "2026-04-01"
WINDOW_END = "2026-05-28"

GROUP_A = [
    "code generation",
    "code completion",
    "code synthesis",
    "program synthesis",
    "code infilling",
    "automated programming",
]

# Local-side Boolean filter: title or abstract must contain at least one term
# from Group B and at least one from Group C (matches the published methodology).
GROUP_B_PATTERNS = [
    r"\befficien\w*", r"\boptimi\w*", r"\baccelerat\w*", r"\blightweight\b",
    r"\bcompress\w*", r"\bquantiz\w*", r"\bprun\w*", r"\bdistill\w*",
    r"\blatency\b", r"\bthroughput\b", r"\bcomputational cost\b",
    r"\benergy\b", r"\bscalab\w*",
]
GROUP_C_PATTERNS = [
    r"\blarge language model", r"\bLLM\b", r"\blanguage model", r"\btransformer",
    r"\bpre-?trained model", r"\bfoundation model", r"\bneural network",
    r"\bdeep learning",
]

GROUP_B_RE = [re.compile(p, re.IGNORECASE) for p in GROUP_B_PATTERNS]
GROUP_C_RE = [re.compile(p, re.IGNORECASE) for p in GROUP_C_PATTERNS]


def s2_search(query, fields="title,abstract,publicationDate,externalIds,venue,authors"):
    """Bulk search Semantic Scholar with pagination, restricted to window."""
    base = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
    all_papers = []
    token = None
    page = 0
    total_reported = None
    while True:
        page += 1
        params = {
            "query": query,
            "publicationDateOrYear": f"{WINDOW_START}:{WINDOW_END}",
            "fields": fields,
            "limit": 1000,
        }
        if token:
            params["token"] = token
        url = base + "?" + urllib.parse.urlencode(params)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=90) as r:
                    payload = json.loads(r.read())
                break
            except Exception as e:
                if attempt < 4:
                    wait = 10 * (attempt + 1)
                    print(f"    page {page} err {e}, retry in {wait}s", file=sys.stderr)
                    time.sleep(wait)
                else:
                    raise
        if total_reported is None:
            total_reported = payload.get("total")
        all_papers.extend(payload.get("data") or [])
        token = payload.get("token")
        if not token:
            break
        time.sleep(2)
    return all_papers, total_reported


def in_window(paper):
    d = paper.get("publicationDate")
    if not d:
        return False
    return WINDOW_START <= d <= WINDOW_END


def passes_boolean_filter(paper):
    title = paper.get("title") or ""
    abstract = paper.get("abstract") or ""
    text = title + " " + abstract
    has_b = any(p.search(text) for p in GROUP_B_RE)
    has_c = any(p.search(text) for p in GROUP_C_RE)
    return has_b and has_c


def main():
    all_hits = {}
    raw_counts = {}
    print("=" * 70)
    print("Semantic Scholar Bulk search per Group A phrase")
    print("=" * 70)
    for a in GROUP_A:
        # S2 query syntax does not support Boolean OR; query just the Group A
        # phrase and apply the full Group B/C filter locally.
        q = f'"{a}"'
        try:
            papers, total = s2_search(q)
            raw_counts[a] = total or len(papers)
            for p in papers:
                if not p.get("paperId"):
                    continue
                all_hits.setdefault(p["paperId"], p)
            print(f"  {a:24s}  raw total={total or len(papers):5d}  cumulative unique={len(all_hits)}")
            time.sleep(3)
        except Exception as e:
            print(f"  {a:24s}  ERROR: {e}", file=sys.stderr)

    (OUT_DIR / "s2_raw.json").write_text(json.dumps(list(all_hits.values()), ensure_ascii=False, indent=2))
    print(f"\nRaw unique S2 records: {len(all_hits)}")

    # Window filter
    in_window_hits = [p for p in all_hits.values() if in_window(p)]
    print(f"Within 2026-04-01..2026-05-28: {len(in_window_hits)}")

    # Group B AND Group C boolean filter
    boolean_pass = [p for p in in_window_hits if passes_boolean_filter(p)]
    print(f"Pass Group B AND Group C boolean filter: {len(boolean_pass)}")

    (OUT_DIR / "s2_filtered.json").write_text(json.dumps(boolean_pass, ensure_ascii=False, indent=2))
    (OUT_DIR / "raw_counts.json").write_text(json.dumps(raw_counts, indent=2))

    # Concise summary
    print("\n--- Filtered candidates ---")
    for p in sorted(boolean_pass, key=lambda x: x.get("publicationDate") or ""):
        arx = (p.get("externalIds") or {}).get("ArXiv", "-")
        doi = (p.get("externalIds") or {}).get("DOI", "-")
        date = p.get("publicationDate") or "?"
        title = (p.get("title") or "")[:90]
        print(f"  {date}  arxiv={arx:14s}  {title}")


if __name__ == "__main__":
    main()
