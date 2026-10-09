import urllib.parse
A=['"code generation"','"code completion"','"code synthesis"','"program synthesis"','"code infilling"','"automated programming"']
B=['efficient','efficiency','optimization','optimize','optimizing','acceleration','accelerate','accelerating','lightweight','compression','compress','quantization','quantized','pruning','pruned','distillation','distilled','latency','throughput','"computational cost"','energy','scalability','scalable']
C=['"large language model"','LLM','"language model"','transformer','"pre-trained model"','"foundation model"','"neural network"','"deep learning"']
def grp(x): return '('+' OR '.join(x)+')'
Q=grp(A)+' AND '+grp(B)+' AND '+grp(C)
FILTER='title_and_abstract.search:'+Q+',from_publication_date:2017-01-01,to_publication_date:2026-03-06'
def url(cursor='*',per=200):
    return 'https://api.openalex.org/works?'+urllib.parse.urlencode({'filter':FILTER,'per-page':per,'cursor':cursor,'mailto':'szw.survey.audit@example.org','select':'id,doi,ids,title,abstract_inverted_index,publication_year,publication_date,primary_location,type,locations'})
if __name__=='__main__': print(Q); print(url(per=1))
