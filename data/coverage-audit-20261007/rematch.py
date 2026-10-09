import json,csv,re,difflib,sys
S={r['sid']:r for r in map(json.loads,open('sample.jsonl'))}
corp=list(csv.DictReader(open('/root/szw/tosem-r1-work/corpus141/primary-studies-141.csv')))
def toks(t): return set(w for w in re.findall(r'[a-z0-9]+',t.lower()) if len(w)>2 and w not in {'the','and','for','with','via','large','language','models','model','code','generation','llm','llms'})
for sid in sys.argv[1:]:
    t=S[sid]['title']; tk=toks(t)
    best=sorted(corp,key=lambda c: -len(tk&toks(c['Title']))/max(1,len(tk|toks(c['Title']))))[:1][0]
    j=len(tk&toks(best['Title']))/max(1,len(tk|toks(best['Title'])))
    print(f"{sid} {j:.2f} | {t[:80]} || {best['ID']} {best['Title'][:80]}")
