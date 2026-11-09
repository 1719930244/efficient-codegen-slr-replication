"""
Cross-dimensional analysis of reporting compliance data.
Joins compliance flags with paper metadata (RQ, year, venue, category)
to produce deeper insights for RQ5.
"""
import json
import csv
from collections import defaultdict, Counter
from itertools import combinations

# ── Load data ──
with open('/home/szw/school/efficient-codegen-survey/data/scripts/07-compliance/final-compliance.json') as f:
    comp = json.load(f)

with open('/home/szw/school/efficient-codegen-survey/data/fine-grained-classification.csv') as f:
    cls_list = list(csv.DictReader(f))

# Build classification lookup by key
cls_by_key = {r['key']: r for r in cls_list}

# ── Join datasets ──
papers = []
for d in comp['details']:
    key = d['key']
    if key not in cls_by_key:
        continue
    meta = cls_by_key[key]
    papers.append({
        'key': key,
        'title': d.get('title', meta['title']),
        'year': int(meta['year']),
        'venue': meta['venue'],
        'rq': meta['new_rq'],
        'categories': meta['primary_categories'],
        'venue_type': 'conference' if meta['venue'] not in ['arXiv preprint', 'arXiv'] else 'preprint',
        'fc': d['fc'], 'lat': d['lat'], 'mem': d['mem'],
        'energy': d['energy'], 'cost': d['cost'],
        'hw': d['hw'], 'model': d['model'], 'serve': d['serve'],
    })

# Check for journal venues
for p in papers:
    v = p['venue'].lower()
    if any(j in v for j in ['transactions', 'journal', 'tosem', 'tse', 'symmetry']):
        p['venue_type'] = 'journal'

METRICS = ['fc', 'lat', 'mem', 'energy', 'cost']
METRIC_NAMES = {
    'fc': 'Correctness',
    'lat': 'Latency',
    'mem': 'Memory',
    'energy': 'Energy',
    'cost': 'Mon. Cost',
}

print(f"Total joined papers: {len(papers)}")
print()

# ══════════════════════════════════════════════
# Analysis 1: Reporting rates by RQ
# ══════════════════════════════════════════════
print("=" * 70)
print("ANALYSIS 1: Reporting Rates by RQ")
print("=" * 70)

rq_groups = defaultdict(list)
for p in papers:
    rq_groups[p['rq']].append(p)

print(f"\n{'RQ':<6} {'N':>4}  {'Correct':>8} {'Latency':>8} {'Memory':>8} {'Energy':>8} {'Cost':>8}")
print("-" * 60)
for rq in sorted(rq_groups.keys()):
    group = rq_groups[rq]
    n = len(group)
    rates = {m: sum(1 for p in group if p[m]) / n * 100 for m in METRICS}
    print(f"{rq:<6} {n:>4}  {rates['fc']:>7.0f}% {rates['lat']:>7.0f}% {rates['mem']:>7.0f}% {rates['energy']:>7.0f}% {rates['cost']:>7.0f}%")

# ══════════════════════════════════════════════
# Analysis 2: Reporting trends by year
# ══════════════════════════════════════════════
print()
print("=" * 70)
print("ANALYSIS 2: Reporting Trends by Year")
print("=" * 70)

year_groups = defaultdict(list)
for p in papers:
    year_groups[p['year']].append(p)

print(f"\n{'Year':<6} {'N':>4}  {'Correct':>8} {'Latency':>8} {'Memory':>8} {'Energy':>8} {'Cost':>8}")
print("-" * 60)
for year in sorted(year_groups.keys()):
    group = year_groups[year]
    n = len(group)
    rates = {m: sum(1 for p in group if p[m]) / n * 100 for m in METRICS}
    print(f"{year:<6} {n:>4}  {rates['fc']:>7.0f}% {rates['lat']:>7.0f}% {rates['mem']:>7.0f}% {rates['energy']:>7.0f}% {rates['cost']:>7.0f}%")

# ══════════════════════════════════════════════
# Analysis 3: Reporting by venue type
# ══════════════════════════════════════════════
print()
print("=" * 70)
print("ANALYSIS 3: Reporting by Venue Type")
print("=" * 70)

vtype_groups = defaultdict(list)
for p in papers:
    vtype_groups[p['venue_type']].append(p)

print(f"\n{'Type':<12} {'N':>4}  {'Correct':>8} {'Latency':>8} {'Memory':>8} {'Energy':>8} {'Cost':>8} {'Hardware':>8} {'Serving':>8}")
print("-" * 80)
for vtype in ['conference', 'journal', 'preprint']:
    if vtype not in vtype_groups:
        continue
    group = vtype_groups[vtype]
    n = len(group)
    metrics_ext = METRICS + ['hw', 'serve']
    rates = {m: sum(1 for p in group if p[m]) / n * 100 for m in metrics_ext}
    print(f"{vtype:<12} {n:>4}  {rates['fc']:>7.0f}% {rates['lat']:>7.0f}% {rates['mem']:>7.0f}% {rates['energy']:>7.0f}% {rates['cost']:>7.0f}% {rates['hw']:>7.0f}% {rates['serve']:>7.0f}%")

# ══════════════════════════════════════════════
# Analysis 4: Metric co-occurrence
# ══════════════════════════════════════════════
print()
print("=" * 70)
print("ANALYSIS 4: Metric Co-occurrence (among efficiency metrics)")
print("=" * 70)

eff_metrics = ['lat', 'mem', 'energy', 'cost']
eff_names = {'lat': 'Latency', 'mem': 'Memory', 'energy': 'Energy', 'cost': 'Cost'}

# How many efficiency metrics does each paper report?
eff_count = Counter()
for p in papers:
    c = sum(1 for m in eff_metrics if p[m])
    eff_count[c] += 1

print("\nNumber of efficiency metrics reported per paper:")
for c in sorted(eff_count.keys()):
    print(f"  {c} metrics: {eff_count[c]} papers ({eff_count[c]/len(papers)*100:.0f}%)")

# Pairwise co-occurrence
print("\nPairwise co-occurrence (papers reporting BOTH metrics):")
print(f"{'Pair':<25} {'Count':>6} {'% of papers':>12}")
print("-" * 45)
for m1, m2 in combinations(eff_metrics, 2):
    both = sum(1 for p in papers if p[m1] and p[m2])
    print(f"  {eff_names[m1]} + {eff_names[m2]:<12} {both:>6} {both/len(papers)*100:>11.1f}%")

# Papers reporting NO efficiency metric at all
no_eff = sum(1 for p in papers if not any(p[m] for m in eff_metrics))
print(f"\nPapers reporting NO efficiency metric: {no_eff} ({no_eff/len(papers)*100:.0f}%)")

# ══════════════════════════════════════════════
# Analysis 5: By technique sub-category
# ══════════════════════════════════════════════
print()
print("=" * 70)
print("ANALYSIS 5: Reporting by Technique Category (top categories)")
print("=" * 70)

cat_groups = defaultdict(list)
for p in papers:
    cats = [c.strip().split('-')[0] for c in p['categories'].split(';')]
    for cat in cats:
        cat_groups[cat].append(p)

# Category name mapping
CAT_NAMES = {
    '3a': 'SpecDec', '3b': 'EarlyExit', '3c': 'NAR', '3d': 'PromptComp',
    '3e': 'ContextPrune', '3f': 'KVCache', '3g': 'PTQ', '3h': 'Pruning',
    '3i': 'Routing', '3j': 'AdaptSample', '3k': 'MultiAgent', '3l': 'CodeSpec',
    '3m': 'SysSrv', '3n': 'CompactRepr', '3o': 'CacheMgmt',
    '2a': 'EfficientPT', '2b': 'PEFT', '2c': 'Distill', '2d': 'RL4Train', '2f': 'Curriculum',
    '1a': 'DataSelect', '1b': 'DataQuality',
    '4a': 'Benchmark', '4b': 'Metrics',
}

print(f"\n{'Category':<14} {'N':>3}  {'Correct':>8} {'Latency':>8} {'Memory':>8} {'Energy':>8} {'Cost':>8}")
print("-" * 65)
for cat in sorted(cat_groups.keys()):
    if cat not in CAT_NAMES:
        continue
    group = cat_groups[cat]
    n = len(group)
    if n < 3:
        continue
    rates = {m: sum(1 for p in group if p[m]) / n * 100 for m in METRICS}
    print(f"{cat} {CAT_NAMES[cat]:<10} {n:>3}  {rates['fc']:>7.0f}% {rates['lat']:>7.0f}% {rates['mem']:>7.0f}% {rates['energy']:>7.0f}% {rates['cost']:>7.0f}%")

# ══════════════════════════════════════════════
# Analysis 6: RQ3 (Inference) vs RQ4 (Deployment) detailed
# ══════════════════════════════════════════════
print()
print("=" * 70)
print("ANALYSIS 6: RQ3 vs RQ4 - Inference vs Deployment reporting")
print("=" * 70)

# Map papers to the new 5-RQ scheme
# RQ3 inference cats: 3a,3b,3c,3f,3d,3e,3l,3n,3o,3j,3k
# RQ4 deploy cats: 3g,3h,3i,3m
rq3_cats = {'3a', '3b', '3c', '3f', '3d', '3e', '3l', '3n', '3o', '3j', '3k'}
rq4_cats = {'3g', '3h', '3i', '3m'}

rq3_papers = []
rq4_papers = []
for p in papers:
    cats = {c.strip().split('-')[0] for c in p['categories'].split(';')}
    if cats & rq3_cats:
        rq3_papers.append(p)
    if cats & rq4_cats:
        rq4_papers.append(p)

for label, group in [("RQ3 Inference", rq3_papers), ("RQ4 Deployment", rq4_papers)]:
    n = len(group)
    rates = {m: sum(1 for p in group if p[m]) / n * 100 for m in METRICS + ['hw', 'serve']}
    print(f"\n{label} (n={n}):")
    for m in METRICS + ['hw', 'serve']:
        name = METRIC_NAMES.get(m, m)
        print(f"  {name:<15}: {rates[m]:.0f}%")

# ══════════════════════════════════════════════
# Summary: Key findings for paper
# ══════════════════════════════════════════════
print()
print("=" * 70)
print("SUMMARY: Key findings for RQ5 expansion")
print("=" * 70)

# Finding: reporting gap varies by RQ
print("\n1. REPORTING GAP BY RQ:")
for rq in sorted(rq_groups.keys()):
    group = rq_groups[rq]
    n = len(group)
    lat_rate = sum(1 for p in group if p['lat']) / n * 100
    mem_rate = sum(1 for p in group if p['mem']) / n * 100
    print(f"   {rq} (n={n}): latency={lat_rate:.0f}%, memory={mem_rate:.0f}%")

# Finding: year trend
print("\n2. YEAR TREND (latency reporting):")
for year in sorted(year_groups.keys()):
    group = year_groups[year]
    n = len(group)
    if n >= 3:
        lat_rate = sum(1 for p in group if p['lat']) / n * 100
        print(f"   {year} (n={n}): latency={lat_rate:.0f}%")

# Finding: most papers report 0 or 1 efficiency metric
print(f"\n3. EFFICIENCY METRIC DEPTH:")
print(f"   0 metrics: {eff_count.get(0,0)} papers ({eff_count.get(0,0)/len(papers)*100:.0f}%)")
print(f"   1 metric:  {eff_count.get(1,0)} papers ({eff_count.get(1,0)/len(papers)*100:.0f}%)")
print(f"   2+ metrics: {sum(v for k,v in eff_count.items() if k>=2)} papers ({sum(v for k,v in eff_count.items() if k>=2)/len(papers)*100:.0f}%)")
