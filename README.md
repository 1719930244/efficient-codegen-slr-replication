# Replication Package: Towards Efficient LLM-Based Code Generation

Replication package for the systematic literature review: *"Towards Efficient LLM-Based Code Generation: A Systematic Review"* (ACM TOSEM, manuscript TOSEM-2026-0589, revision submitted October 2026). Package version **v2.0** (2026-10-09), see [CHANGELOG.md](CHANGELOG.md).

## Overview

This review systematically analyzes **141 primary studies** on efficiency techniques for LLM-based code generation, organized across **six research questions** spanning the full lifecycle from data preparation through deployment and evaluation, plus a cross-cutting controlled experiment on technique composition. The corpus comes from a single search campaign with a literature cut-off of **2026-08-31**: 89 full-text includes and 36 snowballing/targeted additions (net of 3 duplicates) give 122 studies, and the Phase-4 OpenAlex/arXiv window search for April to August 2026 contributes 19 more (16 OpenAlex plus 6 arXiv minus 3 overlaps).

| RQ | Scope | Studies |
|----|-------|---------|
| RQ1 | Data Preparation (selection, quality, synthesis; distillation and code-specific work counted here) | 29 |
| RQ2 | Model Training (pre-training, PEFT, curriculum, RL) | 26 |
| RQ3 | Inference Optimization (decoding, compression, sampling, orchestration) | 63 |
| RQ4 | Deployment (quantization, pruning, routing, system-level serving) | 28 |
| RQ5 | Evaluation (benchmarks, metrics, reporting compliance) | 27 |
| RQ6 | Technique Composition and Interaction Effects (controlled experiments) | — |

> Note: 32 studies span multiple RQs, so per-RQ counts sum to more than 141. The canonical stage marginals are computed by the mapping documented in `scripts/plot_upset.py` (codes 2c and 3l attach to the data stage; 3g/3h/3i/3m to deployment; 4a/4b to evaluation), which reproduces 29/26/63/28/27 exactly. The legacy `RQ` column of earlier package releases was an incomplete annotation and has been regenerated from this mapping; `classification-scheme.csv`'s RQ column keeps the taxonomy's chapter home for each category. RQ6 is answered by controlled experiments rather than primary-study classification (see `experiments/`).

## Repository Structure

```
data/
  primary-studies.csv          # All 141 primary studies: metadata, canonical RQ column, Quality Tier
  classification-scheme.csv    # Taxonomy: 24 categories, study counts and keys regenerated for 141
  statistics.json              # Summary statistics (year, RQ, category distributions, N=141)
  reporting-compliance.json    # Per-study QA scores (QA1-QA4, all 141 studies)
  search-log.csv               # Structured log of every search executed (main campaign, window, gap)
  venues/
    venue-tier-141.csv         # Per-study venue type and CCF/CORE quality tier (all 141)
    appendix-b-venue-tier.csv  # 122-era tier table kept for provenance
  by-rq/
    rq1-studies.csv ... rq5-studies.csv   # Per-RQ subsets (29/26/63/28/27, full column schema)
  corpus-merge-20261001/       # The 19 window-search studies: per-study extraction records
                               # (categories, QA, 8-item audit), the 122-era inputs, and
                               # recompute.py which regenerates primary-studies-141 byte-identically
  window-search-2026/          # Phase-4 arXiv pass (2026-09-29): funnel, per-record screening
                               # log SCREENING.md, batch record, as-run search script
  monthly-updates/             # 2026-05 and 2026-09 batch records, arXiv round artifacts,
                               # UPDATE-LOG.md (living-update ledger)
  gap-search-20261007/         # March 7-31 2026 gap search: funnels, nominations, dual
                               # screening, adjudication (0 eligible, 1 pending full text)
  coverage-audit-20261007/     # Independent OpenAlex recall audit: 400-record sample, both
                               # screeners' decisions, adjudication, estimates (recall 0.78,
                               # 95% CI 0.54-0.91), as-run scripts
  EXPERIMENT-DATA.md           # Provenance map for all raw experiment data (read this first)
  v100-original-results/       # V100 campaign: aggregates, regeneration, April logs, manifests
  a800-results/                # A800 extension campaign: full raw data of both rental servers
  xgpu-3090ti-217/             # RTX 3090 Ti third-architecture reference point + judged outputs
  lcb-results-216/             # LiveCodeBench contamination-free layer, lab V100 server
  v100-sept-round-216/         # Sep 2026 replication round: acceptance loop, 32B cells,
                               # LCB-on-3090Ti, BCB-instruct-on-V100, 1024-token sensitivity
docs/
  search-queries.md            # Full search protocol: term groups, per-database queries verbatim
  UPDATE-PROCESS.md            # Monthly companion-website update SOP
  REQUIREMENTS.md              # Companion website design notes (historical)
experiments/                   # RQ6 empirical experiments: design doc, as-run scripts, env lock
scripts/
  plot_upset.py                # Regenerates the RQ-intersection figure from primary-studies.csv
  sensitivity_findings.py      # Section 3.8 peer-reviewed-subset counts (reproduces Tables 3-4)
  plot_pareto_a800_cleanmem.py # Regenerates the A800 Pareto figure with clean memory readings
  monthly_update_openalex.py   # Standard monthly living-update search (OpenAlex)
  monthly_update_search.py     # May 2026 batch search (Semantic Scholar bulk, historical)
  monthly_update_classify.py   # Monthly batch regex bucketing (historical)
  generate-data.py             # DEPRECATED: regenerated the v1 data files from screening
                               # intermediates that were not archived; statistics.json and
                               # classification-scheme.csv are now regenerated directly from
                               # primary-studies.csv
figures/                       # Reference renders produced by the plotting scripts
TAXONOMY.md                    # Full classification taxonomy with descriptions
```

## Classification Taxonomy

The taxonomy follows the LLM lifecycle with 24 fine-grained categories. Canonical stage assignment (used for all reported marginals): RQ1 = 1a, 1b, 2c, 3l; RQ2 = 2a, 2b, 2d, 2f; RQ3 = 3a, 3b, 3c, 3d, 3e, 3f, 3j, 3k, 3n, 3o; RQ4 = 3g, 3h, 3i, 3m; RQ5 = 4a, 4b. Distillation (2c) and code-specific optimization (3l) attach to the data stage for counting and are discussed in the chapters of the stages their methods target. See `TAXONOMY.md` for descriptions and `data/classification-scheme.csv` for study-to-category assignments.

## Reporting Compliance Audit (N=141)

All 141 primary studies were audited against an 8-item reporting checklist (Table 5 of the manuscript). Aggregate coverage: functional correctness 95%, model name/version 96%, hardware 68%, latency/throughput 47%, memory 21%, serving configuration 40%, monetary cost 21%, energy/carbon 8%. Reporting depth: 25% of studies report no efficiency metric, 33% exactly one, 43% two or more; only 16% report both latency and memory.

Provenance note, recorded as fact: the per-study 8-item audit sheets of the original 122 studies were not archived; their aggregate counts were back-calculated from the published percentages during the revision audit (`data/corpus-merge-20261001/recompute.py` documents the procedure and its ambiguity bounds). The 19 merged studies carry full per-study audit records (`audit-A.json`, `audit-B.json`). Per-study QA scores (QA1-QA4) for all 141 studies are in `data/reporting-compliance.json`.

## Quality Tiers

`data/venues/venue-tier-141.csv` lists the venue type and quality tier of every study: CCF 2022 catalogue first (A/B/C), then CORE A* for venues outside the CCF catalogue (ICLR only in this corpus), workshops and unlisted venues as unranked, preprints as n/a. Distribution of the 55 peer-reviewed studies: 27 CCF-A, 10 CCF-B, 6 CCF-C, 1 CORE-A*, 11 unranked. The same column is in `primary-studies.csv`.

## RQ6: Empirical Composition and Pareto Experiments

Three controlled experiments evaluate efficiency techniques in combination, something primary-study classification alone cannot answer:

1. **Factorial Composition** (12 configurations on Qwen2.5-Coder-7B-Instruct): crosses quantization (FP16/INT8/INT4) x decoding (standard/speculative) x sampling (greedy/adaptive) and measures interaction effects on pass@1, latency, memory, throughput, and energy.
2. **Pareto Frontier** (15 configurations): 5 model scales (0.5B/1.5B/3B/7B/14B) x 3 precisions (FP16/INT8/INT4) under greedy decoding to map the efficiency-quality trade-off space.
3. **Energy Round** (27 configurations, single pass): NVML `nvmlDeviceGetTotalEnergyConsumption`-based energy measurement for every configuration, validated on V100 driver 570+.

Original campaign hardware: 4x NVIDIA Tesla V100-SXM2-32GB, PyTorch 2.5.1 + CUDA 12.4. All scripts and the design document are in `experiments/`.

The extended suite (2026) adds: five-run A800 replication of the composition and Pareto cells at six scales including 32B; EvalPlus re-judgment (HumanEval+) of every central cell on V100, A800, and a consumer RTX 3090 Ti third-architecture reference point; MBPP (five runs) and BigCodeBench (four runs, plus an instruct-protocol check on the lab V100-PCIE server); a contamination-free LiveCodeBench slice (288 post-release problems); pre-registered hypothesis tests H1-H5; a second model family (DeepSeek-Coder-6.7B-Instruct) at the four central cells; and instrumented draft-acceptance measurements on three architectures. Raw data, per-platform software stacks, quarantined defective outputs, the dropped MBPP+ layer, and the files lost when the second A800 rental expired are documented in `data/EXPERIMENT-DATA.md`.

## Data Description

### primary-studies.csv (141 rows)

| Column | Description |
|--------|-------------|
| ID | Sequential identifier (S001-S141; S123-S141 are the 19 merged window-search studies) |
| Key | Citation key used in the paper |
| Title | Full paper title |
| Year | Year of first public version (2020-2026) |
| Venue | Publication venue ("arXiv preprint" for preprints) |
| Venue Type | conference / journal / preprint |
| Quality Tier | CCF A / CCF B / CCF C / CORE A* / unranked / n/a (preprints) |
| Source | database (88), snowball-backward (10), snowball-forward (8), search-update (19, the merged Phase-4 window studies), blank (16, legacy rows whose provenance label was not recorded in v1) |
| RQ | Canonical stage assignment from the plot_upset mapping, multiple values separated by "; " |
| Primary Categories | Fine-grained classification codes (separated by "; ") |
| Secondary Categories | Additional classification codes, if any |
| Scope Flags | Scope annotations, if any |
| Brief Rationale | Classification rationale |

### Study Characteristics

- **Year distribution**: 2020 (1), 2022 (1), 2023 (14), 2024 (35), 2025 (52), 2026 (38)
- **Venue types**: 42 conference, 13 journal, 86 preprint (61% preprints; 55 peer-reviewed)
- **Quality tiers** (peer-reviewed subset): 27 CCF-A, 10 CCF-B, 6 CCF-C, 1 CORE-A*, 11 unranked

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
| **Total (before dedup)** | **26,265** |
| **After deduplication** | **22,118** |

Execution note: the Semantic Scholar bulk endpoint was queried twice on 2026-03-06; the working progress log records 864 hits from an earlier partial run, and 1,505 is the final protocol count reported above. The final execution record is with the first author; both counts are documented here and in `data/search-log.csv`.

Phase 4 (window search, publication window 2026-04-01 to 2026-08-31, run 2026-09): OpenAlex 436 candidates to 16 selected (`data/monthly-updates/2026-09.json`), arXiv 1,095 raw to 111 nominations to 6 selected (`data/window-search-2026/`), merged with 3 overlaps removed into 19 studies (`data/corpus-merge-20261001/`). A gap search covering 2026-03-07 to 2026-03-31 (run 2026-10-07) yielded 118 unique candidates, zero strictly eligible studies, and one record pending full-text adjudication (`data/gap-search-20261007/`). An independent OpenAlex coverage audit estimates campaign recall at 0.78 (95% CI 0.54-0.91) with three eligible records found in a 400-record sample (`data/coverage-audit-20261007/`).

## PRISMA Flow

```
26,265 raw records (7 databases, searched 2026-03-06)
    |
    v  Deduplication (4,147 removed)
22,118 unique records
    |
    v  Title & abstract screening (18,308 excluded)
 3,810 candidates
    |
    v  Full-text retrieval: 1,700 of 3,810 (2,110 not retrieved)
    v  Full-text screening, five authors (1,611 excluded)
    89 included
    |
    +-- Snowballing: 2,645 candidates -> 18 included (2,627 excluded)
    +-- Targeted search: +18
    |
    v  Quality assessment, duplicate removal (-3)
  122 studies (frozen v1 corpus)
    |
    +-- Phase 4, OpenAlex and arXiv window (April-August 2026):
    |     OpenAlex 436 -> 16; arXiv 111 -> 6; overlaps -3; +19
    v
  141 primary studies (cut-off 2026-08-31)
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

## Living Updates and the Companion Website

The companion website (https://1719930244.github.io/efficient-codegen-slr-replication/) is updated on a monthly cadence: newly published studies are logged in `data/monthly-updates/` and rendered on the site without re-opening the review or changing the corpus. Process: [`docs/UPDATE-PROCESS.md`](docs/UPDATE-PROCESS.md); ledger: `data/monthly-updates/UPDATE-LOG.md`. The May and September 2026 tracker batches predate the corpus merge; the September batch's selected studies (plus the arXiv pass, minus overlaps) are the 19 studies merged into the 141 corpus.

## Known Gaps (recorded honestly)

1. **Stage-1 classifier artifacts** (code, threshold, term weights, ranked list of the 22,118 unique records): exist with the first author and will be added to this package; the response letter carries the corresponding placeholder until then.
2. **Full-text screening decisions** (the 1,700 retrieved records with INCLUDE/EXCLUDE and criterion codes): the original assessment worksheets were on the screening working directory that was not archived off the retired rental servers; reconstruction from the ranked candidate list is planned together with item 1.
3. **Gap-search record G000** (EAAI 2026, DOI 10.1016/j.engappai.2026.114360): full text requires institutional access; its IC4 adjudication is pending.
4. Per-study 8-item audit sheets of the original 122 studies: not archived; see the provenance note under Reporting Compliance Audit.

## License

This replication package is released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
