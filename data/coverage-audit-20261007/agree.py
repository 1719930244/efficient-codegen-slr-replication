import json,collections
A={r['sid']:r for r in map(json.loads,open('screenA.jsonl'))}
B={r['sid']:r for r in map(json.loads,open('screenB.jsonl'))}
S={r['sid']:r for r in map(json.loads,open('sample.jsonl'))}
cats=['include','exclude','uncertain']
n=len(S); agree=sum(A[s]['decision']==B[s]['decision'] for s in S)
pa={c:sum(A[s]['decision']==c for s in S)/n for c in cats}; pb={c:sum(B[s]['decision']==c for s in S)/n for c in cats}
pe=sum(pa[c]*pb[c] for c in cats); po=agree/n; k=(po-pe)/(1-pe)
# binary: include-or-uncertain vs exclude
bi=lambda d: 'keep' if d!='exclude' else 'exclude'
po2=sum(bi(A[s]['decision'])==bi(B[s]['decision']) for s in S)/n
qa=sum(bi(A[s]['decision'])=='keep' for s in S)/n; qb=sum(bi(B[s]['decision'])=='keep' for s in S)/n
pe2=qa*qb+(1-qa)*(1-qb); k2=(po2-pe2)/(1-pe2)
print('A',collections.Counter(A[s]['decision'] for s in S)); print('B',collections.Counter(B[s]['decision'] for s in S))
print('3-way agree %.3f kappa %.3f | binary agree %.3f kappa %.3f'%(po,k,po2,k2))
adj=[s for s in S if A[s]['decision']!='exclude' or B[s]['decision']!='exclude']
print('to adjudicate (any non-exclude):',len(adj))
json.dump({'n':n,'agree3':po,'kappa3':k,'agree_bin':po2,'kappa_bin':k2},open('agreement.json','w'),indent=1)
with open('adjudicate.md','w') as f:
    for s in adj:
        r=S[s]; f.write(f"### {s} | A={A[s]['decision']}({A[s].get('criterion')}) B={B[s]['decision']}({B[s].get('criterion')})\n{r['title']} | {r['year']} | {r['venue']} | {r['type']} | doi:{r['doi']} arxiv:{r['arxiv']}\nA: {A[s].get('reason')}\nB: {B[s].get('reason')}\nABS: {r['abstract'][:1800]}\n\n")
