import json,math
adj=json.load(open('adjudication.json')); n=400; pool=3792; C=99
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); return (c-h)/d,(c+h)/d
out={}
for name,k in [('strict',len(adj['eligible_confirmed_abstract'])),
               ('abstract_level',len(adj['eligible_confirmed_abstract'])+len(adj['eligible_pending_fulltext'])),
               ('inclusive',len(adj['eligible_confirmed_abstract'])+len(adj['eligible_pending_fulltext'])+len(adj['borderline_author_judgment']))]:
    lo,hi=wilson(k,n); p=k/n
    out[name]={'k':k,'p':p,'ci':[lo,hi],'missing':p*pool,'missing_ci':[lo*pool,hi*pool],'recall':C/(C+p*pool),'recall_ci':[C/(C+hi*pool),C/(C+lo*pool)]}
    print(f"{name:15s} k={k:2d} p={p:.4f} [{lo:.4f},{hi:.4f}] missing={p*pool:.0f} [{lo*pool:.0f},{hi*pool:.0f}] recall={C/(C+p*pool):.2f} [{C/(C+hi*pool):.2f},{C/(C+lo*pool):.2f}]")
json.dump(out,open('estimates.json','w'),indent=1)
