# Search Protocol and Queries

This document records the complete search protocol of the review: the term
groups, the exact per-database queries, the executed counts, and the
supplementary window searches. The structured log is
[`data/search-log.csv`](../data/search-log.csv).

## Term groups

The database query combines two term groups with Boolean AND. A third group
characterizes LLM-based approaches; because that terminology varies across
venues and years, it is not part of the database query. It is enforced at
Stage-1 screening through inclusion criterion IC2, together with the
publication-year window of 2017 onward. This keeps the database query
deliberately broad so that no relevant study is missed.

**Group A (code generation tasks, 6 terms):**

```
("code generation" OR "code completion" OR "code synthesis" OR "program synthesis" OR "code infilling" OR "automated programming")
```

**Group B (efficiency concepts, 13 terms):**

```
("efficien*" OR "optimi*" OR "accelerat*" OR "lightweight" OR "compress*" OR "quantiz*" OR "prun*" OR "distill*" OR "latency" OR "throughput" OR "computational cost" OR "energy" OR "scalab*")
```

**Group C (LLM scope, 8 terms, applied as the IC2 screening filter):**

```
("large language model" OR "LLM" OR "language model" OR "transformer" OR "pre-trained model" OR "foundation model" OR "neural network" OR "deep learning")
```

## Main campaign (executed 2026-03-06)

Four databases were searched through their advanced-search interfaces with
the field syntax below; three sources were searched through their APIs, which
limit Boolean querying over abstracts, so each was queried per Group A phrase
and the Group B conjunction was applied as a local filter on the metadata the
source exposes (DBLP exposes titles only).

### IEEE Xplore (7,032 records)

Advanced search, Command Search:

```
("All Metadata":"code generation" OR "All Metadata":"code completion" OR "All Metadata":"code synthesis" OR "All Metadata":"program synthesis" OR "All Metadata":"code infilling" OR "All Metadata":"automated programming") AND ("All Metadata":"efficien*" OR "All Metadata":"optimi*" OR "All Metadata":"accelerat*" OR "All Metadata":"lightweight" OR "All Metadata":"compress*" OR "All Metadata":"quantiz*" OR "All Metadata":"prun*" OR "All Metadata":"distill*" OR "All Metadata":"latency" OR "All Metadata":"throughput" OR "All Metadata":"computational cost" OR "All Metadata":"energy" OR "All Metadata":"scalab*")
```

### ACM Digital Library (6,708 records)

Advanced search over The ACM Guide to Computing Literature:

```
[[All: "code generation"] OR [All: "code completion"] OR [All: "code synthesis"] OR [All: "program synthesis"] OR [All: "code infilling"] OR [All: "automated programming"]] AND [[All: efficien*] OR [All: optimi*] OR [All: accelerat*] OR [All: lightweight] OR [All: compress*] OR [All: quantiz*] OR [All: prun*] OR [All: distill*] OR [All: latency] OR [All: throughput] OR [All: "computational cost"] OR [All: energy] OR [All: scalab*]]
```

### Scopus (7,529 records)

```
TITLE-ABS-KEY("code generation" OR "code completion" OR "code synthesis" OR "program synthesis" OR "code infilling" OR "automated programming") AND TITLE-ABS-KEY(efficien* OR optimi* OR accelerat* OR lightweight OR compress* OR quantiz* OR prun* OR distill* OR latency OR throughput OR "computational cost" OR energy OR scalab*)
```

### Web of Science (805 records)

```
TS=("code generation" OR "code completion" OR "code synthesis" OR "program synthesis" OR "code infilling" OR "automated programming") AND TS=(efficien* OR optimi* OR accelerat* OR lightweight OR compress* OR quantiz* OR prun* OR distill* OR latency OR throughput OR "computational cost" OR energy OR scalab*)
```

### Semantic Scholar (1,505 records)

Bulk-search API, one query per Group A phrase; a record was kept when its
title and abstract matched Group B. Execution note, recorded as fact: the
working progress log of 2026-03-06 lists 864 hits from an earlier partial API
run on the same day; the final protocol count reported by the review is
1,505. The final execution record is with the first author; both counts are
documented here.

### arXiv (2,197 records)

API, one `all:<phrase>` query per Group A phrase in categories `cs.*`, with
the Group B conjunction applied locally on title and abstract.

### DBLP (489 records)

API, one title query per Group A phrase, with the Group B filter applied
locally on the title.

### Totals

26,265 raw records; 22,118 unique records after deduplication (Figure 2 of
the manuscript). No date restriction was applied at query time; the 2017
onward window was enforced at Stage-1 screening.

## Window search, April to August 2026 (Phase 4 of the campaign)

The same two-group query schema was reused on two openly indexable sources
for the publication window 2026-04-01 to 2026-08-31. IEEE Xplore, ACM DL,
Scopus, and Web of Science require institutional credentials and are covered
by the annual full sweep.

- **OpenAlex pass (run 2026-09-07**, execution window through 2026-09-07**)**:
  `scripts/monthly_update_openalex.py` schema. 436 candidates, 25 nominated,
  23 unique, 16 selected. Record: `data/monthly-updates/2026-09.json`.
- **arXiv pass (run 2026-09-29)**: `data/window-search-2026/arxiv_search.py`.
  1,095 raw, 413 boolean, 127 title-strict, 111 nominated new, 6 selected.
  Full per-record screening log: `data/window-search-2026/SCREENING.md`;
  batch record: `data/window-search-2026/batch-2026-08-31.json`.
- **Merge (2026-10-01)**: 16 + 6 with 3 overlaps removed = 19 studies merged
  into the corpus under the single-campaign cut-off of 2026-08-31, giving the
  141-study corpus. Per-study extraction records (categories, QA scores,
  8-item reporting audit) and the merge recomputation script:
  `data/corpus-merge-20261001/`.

## Gap search, 7 to 31 March 2026

The main campaign ran on 2026-03-06 and the window search starts 2026-04-01,
leaving 25 days uncovered. The gap was closed on 2026-10-07 with the same
scripts and criteria: OpenAlex 5,954 Group A hits, 561 boolean, 116
title-strict; arXiv 150 raw, 54 boolean, 12 title-strict, 12 nominated new;
118 unique candidates after merge, zero overlap with the corpus. Dual
automated screening agreed on 99 of 118; the 38 records not excluded by both
screeners were adjudicated record by record. Result: zero strictly eligible
studies, one record pending full-text adjudication (G000, EAAI 2026,
DOI 10.1016/j.engappai.2026.114360, abstract not publicly available).
Artifacts: `data/gap-search-20261007/`.

## Coverage audit (2026-10-07)

An independent OpenAlex-based audit estimated the campaign's recall at
approximately 0.78 (95% CI 0.54 to 0.91) against a 400-record sample.
Sample, both screeners' per-record decisions, adjudication, and estimation
scripts: `data/coverage-audit-20261007/`.
