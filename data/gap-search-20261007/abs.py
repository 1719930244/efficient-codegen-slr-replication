import json,urllib.request,urllib.parse,sys,time
sys.path.insert(0,'/home/szw/tosem-survey-hub/efficient-codegen-slr-replication/scripts')
import monthly_update_openalex as m
G='/root/szw/tosem-r1-work/gap-0307-0331'
oa=json.load(open(G+'/openalex/nominated.json'))
ids=[w['openalex_id'].split('/')[-1] for w in oa]
absd={}
for i in range(0,len(ids),50):
    q=urllib.parse.urlencode({'filter':'openalex_id:'+'|'.join(ids[i:i+50]),'per-page':50,'mailto':m.MAILTO,'select':'id,abstract_inverted_index,type,primary_location'})
    d=json.load(urllib.request.urlopen(m.API+'?'+q,timeout=60))
    for w in d['results']: absd[w['id']]=(m.abstract_from_index(w.get('abstract_inverted_index')),w.get('type'))
    time.sleep(1)
rest=json.load(open(G+'/to-screen.json'))
byid={w['openalex_id']:w for w in oa}
for v in rest:
    if v['src'].startswith('openalex'):
        oid=[w['openalex_id'] for w in oa if w['title']==v['title']][0]
        a,t=absd.get(oid,('',None)); v['abstract']=v['abstract'] or a or ''; v['type']=t
        v['venue']=byid[oid].get('venue')
json.dump(rest,open(G+'/to-screen.json','w'),indent=1,ensure_ascii=False)
print(len(absd),'fetched;',sum(1 for v in rest if not v['abstract']),'still without abstract')
