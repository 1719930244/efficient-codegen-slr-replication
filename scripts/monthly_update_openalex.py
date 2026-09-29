#!/usr/bin/env python3
"""Monthly update search via OpenAlex.

Parameterized successor of monthly_update_search.py (Semantic Scholar bulk,
2026-05 batch). The September 2026 batch moved to OpenAlex because the S2
bulk endpoint was rate-limited; this script makes that path reproducible and
is the standard entry point for every monthly batch from October 2026 on.

Replicates the structured search of sec-methodology.tex for an arbitrary
publication-date window:

  Group A (code-generation phrases) as OpenAlex title_and_abstract.search
  queries, then a local Boolean filter requiring at least one Group B
  (efficiency) and one Group C (LLM) pattern in title or abstract, matching
  the published methodology. A title-strict nomination list additionally
  requires the Group B efficiency signal in the title itself.

IEEE Xplore, ACM Digital Library, Scopus, and Web of Science still require
institutional credentials and remain deferred to the annual full sweep.

Usage:
  python scripts/monthly_update_openalex.py --start 2026-09-08 --end 2026-10-01 \
      --outdir data/monthly-update-2026-10

Outputs (in --outdir):
  candidates-raw.json     every Group A hit in the window (merged, deduplicated)
  candidates-boolean.json after the local B AND C filter
  nominated.json          title-strict subset (Group B signal in title)
  funnel.json             stage counts for the batch record

Deduplication against the frozen 122-study corpus and all previously logged
monthly batches happens here by normalized title, so nominated.json contains
genuinely new studies only.
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

MAILTO = "52708958+1719930244@users.noreply.github.com"  # OpenAlex polite pool
API = "https://api.openalex.org/works"
PER_PAGE = 200
MAX_RETRIES = 6

GROUP_A = [
    "code generation",
    "code completion",
    "code synthesis",
    "program synthesis",
    "code infilling",
    "automated programming",
]

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
GROUP_B_TITLE_RE = re.compile("|".join(GROUP_B_PATTERNS), re.IGNORECASE)


def norm_title(t):
    return re.sub(r"[^a-z0-9 ]", "", (t or "").lower()).strip()


def abstract_from_index(inv):
    """Reconstruct abstract text from an OpenAlex inverted index."""
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def fetch_page(search, start, end, cursor):
    params = {
        "filter": (
            f"title_and_abstract.search:{search},"
            f"from_publication_date:{start},to_publication_date:{end}"
        ),
        "per-page": str(PER_PAGE),
        "cursor": cursor,
        "mailto": MAILTO,
        "select": "id,doi,title,display_name,publication_date,primary_location,abstract_inverted_index",
    }
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "efficient-codegen-slr-monthly-update/1.0"}
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            code = getattr(e, "code", None)
            wait = 5 * (attempt + 1) if code in (429, 500, 502, 503) or code is None else 10
            print(f"  retry {attempt + 1}/{MAX_RETRIES} after {wait}s ({e})", file=sys.stderr)
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(wait)


def search_group_a(start, end):
    merged = {}
    for phrase in GROUP_A:
        cursor = "*"
        hits = 0
        while cursor:
            page = fetch_page(phrase, start, end, cursor)
            for w in page.get("results", []):
                wid = w.get("id")
                if wid and wid not in merged:
                    merged[wid] = w
            hits += len(page.get("results", []))
            cursor = (page.get("meta", {}) or {}).get("next_cursor")
            time.sleep(0.2)
        print(f"  [{phrase}] {hits} hits", file=sys.stderr)
    return list(merged.values())


def known_titles(repo_root):
    """Normalized titles of the frozen corpus plus every prior monthly batch."""
    known = set()
    csv_path = repo_root / "data" / "primary-studies.csv"
    if csv_path.exists():
        import csv
        with open(csv_path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                t = row.get("Title") or row.get("title")
                if t:
                    known.add(norm_title(t))
    for jf in sorted((repo_root / "data" / "monthly-updates").glob("*.json")):
        try:
            data = json.load(open(jf, encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for s in data.get("studies", []):
            if s.get("title"):
                known.add(norm_title(s["title"]))
        for d in data.get("dropped", []):
            if isinstance(d, str):
                known.add(norm_title(d))
            elif isinstance(d, dict) and d.get("title"):
                known.add(norm_title(d["title"]))
    return known


def slim(w):
    loc = (w.get("primary_location") or {}).get("source") or {}
    return {
        "openalex_id": w.get("id"),
        "doi": w.get("doi"),
        "title": w.get("title") or w.get("display_name"),
        "publication_date": w.get("publication_date"),
        "venue": loc.get("display_name"),
        "venue_type": loc.get("type"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True, help="window start YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", required=True, help="window end YYYY-MM-DD (inclusive)")
    ap.add_argument("--outdir", required=True, help="output directory for this batch")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = repo_root / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Window {args.start} .. {args.end}", file=sys.stderr)
    raw = search_group_a(args.start, args.end)
    print(f"Group A merged: {len(raw)}", file=sys.stderr)

    known = known_titles(repo_root)
    print(f"Known titles for dedup: {len(known)}", file=sys.stderr)

    boolean, seen = [], set()
    for w in raw:
        title = w.get("title") or w.get("display_name") or ""
        nt = norm_title(title)
        if not nt or nt in known or nt in seen:
            continue
        text = title + " " + abstract_from_index(w.get("abstract_inverted_index"))
        if any(r.search(text) for r in GROUP_B_RE) and any(r.search(text) for r in GROUP_C_RE):
            seen.add(nt)
            boolean.append(slim(w))

    nominated = [c for c in boolean if GROUP_B_TITLE_RE.search(c["title"] or "")]
    nominated.sort(key=lambda c: c["publication_date"] or "")
    boolean.sort(key=lambda c: c["publication_date"] or "")

    (outdir / "candidates-raw.json").write_text(
        json.dumps([slim(w) for w in raw], ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "candidates-boolean.json").write_text(
        json.dumps(boolean, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "nominated.json").write_text(
        json.dumps(nominated, ensure_ascii=False, indent=1), encoding="utf-8")

    funnel = {
        "window": f"{args.start}..{args.end}",
        "group_a_hits": len(raw),
        "boolean_pass": len(boolean),
        "nominated_title_strict": len(nominated),
        "dedup_against": "primary-studies.csv + all data/monthly-updates/*.json (normalized titles)",
    }
    (outdir / "funnel.json").write_text(json.dumps(funnel, indent=1), encoding="utf-8")
    print(json.dumps(funnel, indent=1))
    print(f"Wrote {outdir}", file=sys.stderr)


if __name__ == "__main__":
    main()
