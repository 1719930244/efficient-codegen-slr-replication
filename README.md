# Replication Package: Towards Efficient LLM-Based Code Generation

Replication package for the systematic literature review *"Towards Efficient LLM-Based Code Generation: A Systematic Review"* (ACM TOSEM, manuscript TOSEM-2026-0589, revision, November 2026). Package version **v4.0** (2026-11-24).

## Overview

This review systematically analyzes **116 primary studies** on efficiency techniques for LLM-based code generation, organized across **six research questions** spanning the full lifecycle from data preparation through deployment and evaluation, plus a cross-cutting controlled experiment on technique composition. The literature cut-off is **2026-10-31**. The search returned 43,521 records; deduplication removed 4,147 and left 39,374 unique records; title and abstract screening retained 3,840 candidates; 1,730 full texts were retrieved and screened, which identified 119 included studies. Snowballing and the targeted search added 36 candidates, 3 duplicates were removed, and a confirmation pass re-read all 152 against the criteria and removed 35; merging one study that appears both as an ASE paper and as its arXiv version leaves 116 primary studies.

| RQ | Scope | Studies |
|----|-------|---------|
| RQ1 | Data Preparation (selection, quality, synthesis; distillation and code-specific work counted here) | 23 |
| RQ2 | Model Training (pre-training, PEFT, curriculum, RL) | 20 |
| RQ3 | Inference Optimization (decoding, compression, sampling, orchestration) | 59 |
| RQ4 | Deployment (quantization, pruning, routing, system-level serving) | 26 |
| RQ5 | Evaluation (benchmarks, metrics, reporting compliance) | 15 |
| RQ6 | Technique Composition and Interaction Effects (controlled experiments) | — |

Note: 27 studies span two RQs, so per-RQ counts sum to more than 116. The stage marginals are computed by the mapping in `scripts/plot_upset.py` (codes 2c and 3l attach to the data stage; 3g/3h/3i/3m to deployment; 4a/4b to evaluation), which reproduces 23/20/59/26/15. `classification-scheme.csv`'s RQ column keeps the taxonomy's chapter home for each category. RQ6 is answered by controlled experiments rather than primary-study classification (see `experiments/`).

## Repository Structure

```
data/
  primary-studies.csv          # All 116 primary studies: metadata, RQ, categories, Quality Tier
  classification-scheme.csv    # Taxonomy: 15 groups over 24 fine-grained categories with study counts
  statistics-116.json          # Summary statistics (year, RQ, category, quality-tier distributions)
  reporting-compliance.json    # Per-study QA scores (QA1-QA4) for all 116 studies
  search-log.csv               # Every search executed (main campaign, update searches, gap search)
  venues/venue-tier-116.csv    # Per-study venue type and CCF/CORE quality tier
  by-rq/                       # Per-RQ subsets (23/20/59/26/15)
  reporting-audit/             # Per-study 8-item reporting audit (reporting-audit-116.csv)
  screening-decisions/         # Stage-1 classifier and labels, full-text screening records,
                               # update-search decisions, confirmation-pass exclusions,
                               # coverage audit (see its README)
  window-search-2026/          # April-August 2026 arXiv update search: funnel, per-record
                               # screening log SCREENING.md, batch record, search script
  monthly-updates/             # Update-search batch records rendered on the companion website,
                               # UPDATE-LOG.md
  EXPERIMENT-DATA.md           # Provenance map for all raw experiment data (read this first)
  v100-original-results/       # V100 campaign: aggregates, regeneration, logs, manifests
  a800-results/                # A800 extension campaign: raw data of both servers
  xgpu-3090ti/                 # RTX 3090 Ti third-architecture reference point + judged outputs
  lcb-results/                 # LiveCodeBench contamination-free layer
  v100-sept-round/             # Sep 2026 replication round: acceptance loop, 32B cells,
                               # LCB-on-3090Ti, BCB-instruct-on-V100, 1024-token sensitivity
docs/
  search-queries.md            # Full search protocol: term groups, per-database queries verbatim
  index.html                   # Companion website
experiments/                   # RQ6 experiments: as-run scripts and environment lock
scripts/
  plot_upset.py                # Regenerates the RQ-intersection figure from primary-studies.csv
  aggregate_reporting_audit.py # Regenerates the reporting-audit rates from reporting-audit-116.csv
  sensitivity_findings.py      # Peer-reviewed share of the studies behind each finding
  placeholder_failures.py      # Section 9.1 placeholder answers under the raw completion prompt
  plot_pareto_a800_cleanmem.py # Regenerates the A800 Pareto figure
  monthly_update_openalex.py   # OpenAlex update search (local Group B and title filters)
  monthly_update_search.py     # Semantic Scholar bulk update search (May 2026 batch)
figures/                       # Reference renders produced by the plotting scripts
TAXONOMY.md                    # Full classification taxonomy with descriptions
```

Absolute paths in the raw logs and as-run scripts are shortened to `~/`.

## Classification Taxonomy

The taxonomy follows the LLM lifecycle with 24 fine-grained categories. Stage assignment used for all reported marginals: RQ1 = 1a, 1b, 2c, 3l; RQ2 = 2a, 2b, 2d, 2f; RQ3 = 3a, 3b, 3c, 3d, 3e, 3f, 3j, 3k, 3n, 3o; RQ4 = 3g, 3h, 3i, 3m; RQ5 = 4a, 4b. Distillation (2c) and code-specific optimization (3l) attach to the data stage for counting and are discussed in the chapters of the stages their methods target. See `TAXONOMY.md` for descriptions and `data/classification-scheme.csv` for study-to-category assignments.

## Reporting Audit (N=116)

All 116 studies were audited against an 8-item reporting checklist (Table 3 of the manuscript). Coverage: functional correctness 89%, model name/version 92%, hardware 66%, latency/throughput 50%, memory 16%, serving configuration 69%, monetary cost 16%, energy/carbon 7%. Nineteen studies report no efficiency metric, and only twelve report latency and memory together.

Per-study codings are in `data/reporting-audit/reporting-audit-116.csv`; `scripts/aggregate_reporting_audit.py` regenerates the rates, the breadth counts, and the per-stage table. Full texts were read for 110 studies; the other six were coded from their abstracts and metadata. Per-study QA scores (QA1-QA4) are in `data/reporting-compliance.json`.

## Screening Decisions

`data/screening-decisions/` holds the decision records of every screening step:

- `stage1-classifier/`: the rule-based Stage-1 classifier, the Stage-1 labels of the main search with their matched-term counts, and `verify_labels.py`, which re-derives every label from the rule. The final 3,840 Stage-1 candidates also include records that the authors retained during screening.
- `fulltext-screening/`: the scripted first pass and the automated re-screening of the full texts of the main search. The authors checked every paper that the re-screening retained.
- `window-2026-arxiv.csv` and `monthly-2026-10.csv`: decisions for the update searches of April-August and September-October 2026.
- `final-corpus-116.csv` and `confirmation-exclusions-35.csv`: the final corpus and the 35 studies removed in the confirmation pass, with the failed criterion, the evidence, and the reason.
- `coverage-audit/`: the 400-record coverage-audit sample with both screeners' decisions, the adjudication, and the recall estimate.

## Quality Tiers

`data/venues/venue-tier-116.csv` lists the venue type and quality tier of every study: CCF 2022 catalogue first (A/B/C), then CORE A* for venues outside the CCF catalogue (ICLR only in this corpus), workshops and unlisted venues as unranked, preprints as n/a. Distribution of the 54 peer-reviewed studies: 27 CCF-A, 9 CCF-B, 6 CCF-C, 1 CORE-A*, 11 unranked; the 62 preprints are n/a. The same column is in `primary-studies.csv`.

## RQ6: Empirical Composition and Pareto Experiments

Three controlled experiments evaluate efficiency techniques in combination, something primary-study classification alone cannot answer:

1. **Factorial Composition** (12 configurations on Qwen2.5-Coder-7B-Instruct): crosses quantization (FP16/INT8/INT4) x decoding (standard/speculative) x sampling (greedy/adaptive) and measures interaction effects on Pass@1, latency, memory, throughput, and energy.
2. **Pareto Frontier** (15 configurations): 5 model scales (0.5B/1.5B/3B/7B/14B) x 3 precisions (FP16/INT8/INT4) under greedy decoding to map the efficiency-quality trade-off space.
3. **Energy Round** (27 configurations, single pass): NVML `nvmlDeviceGetTotalEnergyConsumption`-based energy measurement for every configuration, validated on V100 driver 570+.

Original campaign hardware: 4x NVIDIA Tesla V100-SXM2-32GB, PyTorch 2.5.1 + CUDA 12.4. The as-run scripts are in `experiments/`.

The extended suite (2026) adds: five-run A800 replication of the composition and Pareto cells at six scales including 32B; EvalPlus re-judgment (HumanEval+) of every central cell on V100, A800, and a consumer RTX 3090 Ti third-architecture reference point; MBPP (five runs) and BigCodeBench (four runs, plus an instruct-protocol check on a V100-PCIE server); a contamination-free LiveCodeBench slice (288 post-release problems); a second model family (DeepSeek-Coder-6.7B-Instruct) at the four central cells; and instrumented draft-acceptance measurements on three architectures. Raw data, per-platform software stacks, quarantined defective outputs, and the dropped MBPP+ layer are documented in `data/EXPERIMENT-DATA.md`.

## Data Description

### primary-studies.csv (116 rows)

| Column | Description |
|--------|-------------|
| ID | Identifier from the 152-candidate set; identifiers of studies removed in the confirmation pass are absent. S123-S141 are update-search studies of April-August 2026, S142-S152 those of September-October 2026 |
| Key | Citation key used in the paper |
| Title | Full paper title |
| Year | Year of first public version (2022-2026) |
| Venue | Publication venue ("arXiv preprint" for preprints) |
| Venue Type | conference / journal / preprint |
| Quality Tier | CCF A / CCF B / CCF C / CORE A* / unranked / n/a (preprints) |
| Source | database (85), search-update (15, April-August 2026 update search), monthly-update-2026-10 (7, September-October 2026 update search), snowball-backward (3), snowball-forward (3), blank (3, provenance label not recorded) |
| RQ | Stage assignment from the plot_upset mapping, multiple values separated by "; " |
| Primary Categories | Fine-grained classification codes (separated by "; ") |
| Secondary Categories | Additional classification codes, if any |
| Scope Flags | Scope annotations, if any |
| Brief Rationale | Classification rationale |

### Study Characteristics

- **Year distribution**: 2022 (1), 2023 (13), 2024 (28), 2025 (38), 2026 (36)
- **Venue types**: 41 conference, 13 journal, 62 preprint (53% preprints; 54 peer-reviewed)
- **Quality tiers** (peer-reviewed subset): 27 CCF-A, 9 CCF-B, 6 CCF-C, 1 CORE-A*, 11 unranked
- **RQ marginals**: RQ1 23, RQ2 20, RQ3 59, RQ4 26, RQ5 15; 27 studies span two RQs

## Search Strategy

Seven digital libraries were searched on **2026-03-06** with a two-group database query: Group A (6 code-generation phrases) AND Group B (13 efficiency terms). Group C (8 LLM-scope terms) and the 2017-onward year window are eligibility filters applied at Stage-1 screening (inclusion criterion IC2), not part of the database query. Full queries verbatim: [`docs/search-queries.md`](docs/search-queries.md); structured log: [`data/search-log.csv`](data/search-log.csv).

| Database | Records |
|----------|---------|
| Scopus | 7,529 |
| IEEE Xplore | 7,032 |
| ACM Digital Library | 6,708 |
| arXiv | 2,197 |
| Semantic Scholar | 1,505 |
| Web of Science | 805 |
| DBLP | 489 |
| **Seven-library total (main search)** | **26,265** |
| **Update searches, 2026-04 to 2026-10** | **17,256** |
| **Total retrieved** | **43,521** |
| **After deduplication** | **39,374** |

Update searches extended the cut-off to 2026-10-31 on OpenAlex and arXiv: for April-August 2026, OpenAlex gave 16 selections (`data/monthly-updates/2026-09.json`) and arXiv 6 (`data/window-search-2026/`), 19 studies after removing 3 overlaps; for September-October 2026, OpenAlex gave 20 selections, of which 11 were kept (`data/monthly-updates/2026-10.json`, `data/screening-decisions/monthly-2026-10.csv`). A gap search for 2026-03-07 to 2026-03-31 found no eligible study. A coverage audit on OpenAlex estimates recall within the records that the query returns there at 0.78 (95% CI 0.54-0.91) from a 400-record sample (`data/screening-decisions/coverage-audit/`).

## PRISMA Flow

```
43,521 records retrieved (cut-off 2026-10-31)
    26,265 from the seven-library search of 2026-03-06
    17,256 from the update searches of April to October 2026
    |
    v  Deduplication (4,147 removed)
39,374 unique records
    |
    v  Title & abstract screening (35,534 excluded)
 3,840 candidates
    |
    v  Full-text retrieval: 1,730 of 3,840
    v  (2,110 candidates below the retrieval cut were not assessed)
    v  Full-text screening: scripted first pass, automated re-screening,
    v  author check of every paper the re-screening retained (1,611 excluded)
   119 included from full-text screening
    |
    +-- Snowballing: 2,645 candidates -> 18 included (2,627 excluded)
    +-- Targeted search: +18
    +-- Duplicates removed: -3
    |
    v  Quality assessment over 152 candidates
    |
    v  Confirmation pass: 35 removed
    v  (1 IC2, 15 IC4/EC1, 1 EC5, 18 runtime-of-generated-code)
   117 studies
    |
    v  ASE paper and its arXiv version merged under EC5
116 primary studies (cut-off 2026-10-31)
```

## Quality Assessment

Each study was assessed on four criteria; scores are per-study in `data/reporting-compliance.json`:

| Criterion | Description |
|-----------|-------------|
| QA1 | Does the study include a quantitative empirical evaluation? |
| QA2 | Does the study report at least one baseline comparison? |
| QA3 | Does the study disclose the experimental configuration? |
| QA4 | Does the study provide reproducibility artifacts? |

Scores: 1.0 (fully met), 0.5 (partially met), 0.0 (not met). Maximum total: 4.0.

## Companion Website

The companion website (https://1719930244.github.io/efficient-codegen-slr-replication/) lists newly published studies on a monthly cadence without re-opening the review or changing the corpus. Batch records are in `data/monthly-updates/`, with the ledger in `data/monthly-updates/UPDATE-LOG.md`.

## License

This replication package is released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
