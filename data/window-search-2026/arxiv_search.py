"""arXiv fallback of monthly_update_openalex.py for window 2026-04-01..2026-08-31
(OpenAlex daily quota exhausted on 2026-09-29). Same three-group schema:
Group A phrase in title/abstract via arXiv API, local B AND C filter, title-strict
nomination, dedup against frozen corpus + prior batches by normalized title."""
import json, re, sys, time, urllib.parse, urllib.request, csv
import xml.etree.ElementTree as ET
sys.path.insert(0, '/home/szw/tosem-survey-hub/efficient-codegen-slr-replication/scripts')
import monthly_update_openalex as m
from pathlib import Path
NS = {'a': 'http://www.w3.org/2005/Atom'}
OUT = Path('/root/szw/tosem-r1-work/lit-arxiv')
merged = {}
for phrase in m.GROUP_A:
    start = 0
    while True:
        q = f'(ti:"{phrase}" OR abs:"{phrase}") AND submittedDate:[202604010000 TO 202608312359]'
        url = 'http://export.arxiv.org/api/query?' + urllib.parse.urlencode(
            {'search_query': q, 'start': start, 'max_results': 200, 'sortBy': 'submittedDate'})
        for att in range(6):
            try:
                x = urllib.request.urlopen(url, timeout=90).read(); break
            except Exception as e:
                print('retry', e, file=sys.stderr); time.sleep(10 * (att + 1))
        root = ET.fromstring(x)
        ents = root.findall('a:entry', NS)
        for e in ents:
            aid = e.find('a:id', NS).text.split('/abs/')[-1]
            merged.setdefault(aid.split('v')[0], {
                'arxiv': aid, 'title': ' '.join(e.find('a:title', NS).text.split()),
                'abstract': ' '.join(e.find('a:summary', NS).text.split()),
                'published': e.find('a:published', NS).text[:10],
                'categories': [c.get('term') for c in e.findall('a:category', NS)]})
        print(phrase, start, len(ents), file=sys.stderr)
        if len(ents) < 200: break
        start += 200; time.sleep(3.5)
    time.sleep(3.5)
raw = list(merged.values())
boolean = [w for w in raw if any(r.search(w['title'] + ' ' + w['abstract']) for r in m.GROUP_B_RE)
           and any(r.search(w['title'] + ' ' + w['abstract']) for r in m.GROUP_C_RE)]
known = m.known_titles(Path('/home/szw/tosem-survey-hub/efficient-codegen-slr-replication'))
for pp in json.load(open('/home/szw/tosem-survey-hub/efficient-codegen-slr-replication/data/monthly-updates/2026-05.json'))['papers']:
    known.add(m.norm_title(pp['title']))
titled = [w for w in boolean if m.GROUP_B_TITLE_RE.search(w['title'])]
nominated = [w for w in titled if m.norm_title(w['title']) not in known]
already = [w for w in titled if m.norm_title(w['title']) in known]
for n, d in [('candidates-raw', raw), ('candidates-boolean', boolean), ('nominated', nominated), ('title-strict-known', already)]:
    json.dump(d, open(OUT / f'{n}.json', 'w'), indent=1, ensure_ascii=False)
json.dump({'raw': len(raw), 'boolean': len(boolean), 'title_strict': len(titled), 'nominated_new': len(nominated), 'already_known': len(already)}, open(OUT / 'funnel.json', 'w'), indent=1)
print(open(OUT / 'funnel.json').read())
