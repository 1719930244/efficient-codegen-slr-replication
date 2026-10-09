import json,csv,re
G='/root/szw/tosem-r1-work/gap-0307-0331'
def nt(t): return re.sub(r'[^a-z0-9 ]','',(t or '').lower()).strip()
oa=json.load(open(G+'/openalex/nominated.json')); ax=json.load(open(G+'/arxiv/nominated.json'))
corp=set()
for r in csv.DictReader(open('/root/szw/tosem-r1-work/corpus141/primary-studies-141.csv')):
    corp.add(nt(r.get('title') or r.get('Title') or ''))
out={}; 
for w in oa:
    out.setdefault(nt(w['title']),{'src':'openalex','title':w['title'],'date':w.get('publication_date'),'id':w.get('doi') or w.get('id'),'abstract':w.get('abstract') or ''})
for w in ax:
    k=nt(w['title'])
    if k in out: out[k]['src']+='+arxiv'
    else: out[k]={'src':'arxiv','title':w['title'],'date':w['published'],'id':'arXiv:'+w['arxiv'],'abstract':w['abstract']}
inc=[v for k,v in out.items() if k in corp]
rest=[v for k,v in out.items() if k not in corp]
print('oa',len(oa),'ax',len(ax),'union',len(out),'in141',len(inc),'to screen',len(rest))
for v in inc: print('  in corpus:',v['title'][:90])
rest.sort(key=lambda v:v['date'] or '')
for i,v in enumerate(rest): v['gid']=f'G{i:03d}'
json.dump(rest,open(G+'/to-screen.json','w'),indent=1,ensure_ascii=False)
print(sum(1 for v in rest if not v['abstract']),'without abstract')
