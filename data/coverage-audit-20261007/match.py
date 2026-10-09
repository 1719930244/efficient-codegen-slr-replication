import json,csv,re,difflib
H={}
for l in open('hits.jsonl'):
    r=json.loads(l); H[r['openalex']]=r
H=list(H.values())
def norm(t): return re.sub(r'[^a-z0-9 ]','',re.sub(r'\s+',' ',(t or '').lower())).strip()
corp=list(csv.DictReader(open('/root/szw/tosem-r1-work/corpus141/primary-studies-141.csv')))
for c in corp:
    m=re.match(r'arxiv_(\d{4})_(\d{4,5})',c['Key']); c['_arx']=f'{m.group(1)}.{m.group(2)}' if m else ''
    c['_t']=norm(c['Title'])
ht={norm(h['title']):h for h in H}
matched={}; border=[]
for c in corp:
    hit=None
    for h in H:
        if c['_arx'] and h['arxiv'] and h['arxiv'].startswith(c['_arx']): hit=h;break
    if not hit and c['_t'] in ht: hit=ht[c['_t']]
    if not hit:
        best=max(H,key=lambda h: difflib.SequenceMatcher(None,c['_t'],norm(h['title'])).ratio() if abs(len(c['_t'])-len(norm(h['title'])))<40 else 0)
        s=difflib.SequenceMatcher(None,c['_t'],norm(best['title'])).ratio()
        if s>=0.9: hit=best
        elif s>=0.75: border.append((c['ID'],c['Title'],best['title'],round(s,3)))
    if hit: matched[c['ID']]=hit['openalex']
ids=set(matched.values())
pool=[h for h in H if h['openalex'] not in ids]
json.dump(matched,open('corpus-matched.json','w'),indent=1)
with open('pool.jsonl','w') as f:
    for h in pool: f.write(json.dumps(h)+'\n')
print('unique hits',len(H),'corpus matched',len(matched),'pool',len(pool))
for b in border: print('BORDER',b)
