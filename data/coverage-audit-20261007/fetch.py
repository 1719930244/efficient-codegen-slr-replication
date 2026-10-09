import json,urllib.request,time,datetime
from query import url,Q,FILTER
def abstract(inv):
    if not inv: return ''
    pos=[]
    for w,ps in inv.items():
        for p in ps: pos.append((p,w))
    return ' '.join(w for p,w in sorted(pos))
cur='*'; out=open('hits.jsonl','w'); n=0; total=None
while cur:
    for k in range(5):
        try:
            d=json.load(urllib.request.urlopen(url(cur),timeout=120)); break
        except Exception as e:
            print('retry',k,e); time.sleep(5*(k+1))
    total=d['meta']['count']
    for w in d['results']:
        loc=w.get('primary_location') or {}; src=(loc.get('source') or {})
        ids=w.get('ids') or {}
        arx=''
        for l in (w.get('locations') or []):
            lu=(l.get('landing_page_url') or '')
            if 'arxiv.org/abs/' in lu: arx=lu.split('arxiv.org/abs/')[1].split('v')[0]
        out.write(json.dumps({'openalex':w['id'],'doi':(w.get('doi') or '').replace('https://doi.org/','').lower(),'arxiv':arx,
          'title':w.get('title') or '','abstract':abstract(w.get('abstract_inverted_index')),'year':w.get('publication_year'),
          'date':w.get('publication_date'),'venue':src.get('display_name') or '','type':w.get('type')})+'\n'); n+=1
    cur=d['meta'].get('next_cursor')
    if not d['results']: break
out.close()
json.dump({'executed':datetime.datetime.now().isoformat(),'query':Q,'filter':FILTER,'count':total,'downloaded':n},open('query-meta.json','w'),indent=1)
print(total,n)
