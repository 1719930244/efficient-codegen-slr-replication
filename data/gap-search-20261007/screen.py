import sys,json,time,re
sys.path.insert(0,'/root/fse2027/analysis'); import gpt_call
model,outf=sys.argv[1],sys.argv[2]
S=[json.loads(l) for l in open('sample.jsonl')]
SYS="You are a careful systematic-review screener. You judge records strictly against the given eligibility criteria using only the title and abstract. Output strict JSON only."
CRIT="""Eligibility criteria of a systematic review on the efficiency of LLM-based code generation (title-and-abstract stage):
IC1: The paper proposes or evaluates a technique that improves the efficiency of the code generation PROCESS (resources spent on data preparation, training, inference/generation, or deployment/serving of the model), e.g. faster or cheaper generation, smaller/compressed/quantized/distilled code models, data-efficient training, efficient decoding, prompt compression, efficient serving. Papers whose goal is the runtime efficiency of the GENERATED code (faster programs, optimized kernels, lower gas) do NOT satisfy IC1.
IC2: The approach is LLM-based or Transformer-based (not only traditional ML or shallow networks).
IC3: The task is code completion or natural-language-to-code generation (including repository-level code generation, program synthesis by LLMs, hardware description code generation).
EC2: Exclude if the paper targets other SE tasks (defect detection, test generation, code review, vulnerability detection, code search, summarization, translation-only, etc.).
EC3: Exclude if not written in English.
EC4: Exclude surveys, reviews, editorials, theses-only abstracts, short abstracts.
Decision: "include" if IC1, IC2, IC3 plausibly hold and no EC applies; "exclude" otherwise; "uncertain" only if the abstract is missing or genuinely ambiguous.
For each record return {"sid":..., "decision":"include|exclude|uncertain", "criterion":"the decisive criterion, e.g. IC1 or EC2 or ALL-MET", "reason":"one short line"}.
Return a JSON array, one object per record, same order."""
res={}
try: res={r['sid']:r for r in map(json.loads,open(outf))}
except FileNotFoundError: pass
todo=[s for s in S if s['sid'] not in res]
for i in range(0,len(todo),20):
    batch=todo[i:i+20]
    recs='\n\n'.join(f"[{s['sid']}] TITLE: {s['title']}\nTYPE: {s['type']}  VENUE: {s['venue']}  YEAR: {s['year']}\nABSTRACT: {s['abstract'][:2500] or '(no abstract available)'}" for s in batch)
    for k in range(4):
        try:
            txt,_=gpt_call.call(SYS,CRIT+"\n\nRECORDS:\n"+recs,model=model,max_tokens=6000,timeout=300,retries=2)
            m=re.search(r'\[.*\]',txt,re.S); arr=json.loads(m.group(0))
            got={a['sid']:a for a in arr if a.get('sid') in {b['sid'] for b in batch}}
            if len(got)<len(batch): raise ValueError('missing %d'%(len(batch)-len(got)))
            break
        except (Exception,SystemExit) as e:
            print('batch',i,'retry',k,str(e)[:120],flush=True); time.sleep(20*(k+1)); got={}
    with open(outf,'a') as f:
        for sid,a in got.items(): f.write(json.dumps(a)+'\n')
    print(model,i+len(batch),'done',flush=True); time.sleep(8)
