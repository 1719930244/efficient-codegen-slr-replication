# Replication Package: Towards Efficient LLM-Based Code Generation

Replication package for the systematic literature review: *"Towards Efficient LLM-Based Code Generation: A Systematic Review"* (ACM TOSEM, manuscript TOSEM-2026-0589, revision, November 2026). Package version **v2.5** (2026-11-13), see [CHANGELOG.md](CHANGELOG.md).

## Overview

This review systematically analyzes **117 primary studies** on efficiency techniques for LLM-based code generation, organized across **six research questions** spanning the full lifecycle from data preparation through deployment and evaluation, plus a cross-cutting controlled experiment on technique composition. The corpus comes from a single search campaign with a literature cut-off of **2026-10-31**. The search returned 43,521 records; deduplication removed 4,147 and left 39,374 unique records; title and abstract screening retained 3,840 candidates; 1,730 full texts were retrieved and screened, which identified 119 included studies. Snowballing and the targeted search added 36 candidates, 3 duplicates were removed, and a confirmation pass re-read all 152 against the criteria and removed 35, which leaves 117 primary studies.

| RQ | Scope | Studies |
|----|-------|---------|
| RQ1 | Data Preparation (selection, quality, synthesis; distillation and code-specific work counted here) | 24 |
| RQ2 | Model Training (pre-training, PEFT, curriculum, RL) | 20 |
| RQ3 | Inference Optimization (decoding, compression, sampling, orchestration) | 59 |
| RQ4 | Deployment (quantization, pruning, routing, system-level serving) | 26 |
| RQ5 | Evaluation (benchmarks, metrics, reporting compliance) | 15 |
| RQ6 | Technique Composition and Interaction Effects (controlled experiments) | — |

Note: 27 studies span multiple RQs, so per-RQ counts sum to more than 117. The canonical stage marginals are computed by the mapping documented in `scripts/plot_upset.py` (codes 2c and 3l attach to the data stage; 3g/3h/3i/3m to deployment; 4a/4b to evaluation), which reproduces 24/20/59/26/15 exactly. The legacy `RQ` column of earlier package releases was an incomplete annotation and has been regenerated from this mapping; `classification-scheme.csv`'s RQ column keeps the taxonomy's chapter home for each category. RQ6 is answered by controlled experiments rather than primary-study classification (see `experiments/`).

## Repository Structure

```
data/
  primary-studies.csv          # All 117 primary studies: metadata, canonical RQ column, Quality Tier
  classification-scheme.csv    # Taxonomy: 15 groups over 24 fine-grained categories, study counts and keys regenerated for 117
  statistics-117.json          # Summary statistics (year, RQ, category, quality-tier distributions, N=117); statistics-152.json keeps the pre-confirmation set
  reporting-compliance.json    # Per-study QA scores (QA1-QA4) for all 117 studies
  search-log.csv               # Structured log of every search executed (main campaign, window, gap)
  venues/
    venue-tier-117.csv         # Per-study venue type and CCF/CORE quality tier (all 117); venue-tier-152.csv keeps the pre-confirmation set
    appendix-b-venue-tier.csv  # 122-era tier table kept for provenance
  by-rq/
    rq1-studies.csv ... rq5-studies.csv   # Per-RQ subsets (24/20/59/26/15, full column schema)
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
  placeholder_failures.py      # Section 9.1 placeholder answers under the raw completion prompt
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

## Reporting Compliance Audit (N=117) and QA Coverage (N=117)

All 117 studies were audited against an 8-item reporting checklist (Table 5 of the manuscript). Aggregate coverage: functional correctness 88%, model name/version 91%, hardware 66%, latency/throughput 50%, memory 16%, serving configuration 68%, monetary cost 16%, energy/carbon 7%. Twenty studies report no efficiency metric, and only twelve report latency and memory together.

Per-study codings are in `data/reporting-audit/reporting-audit-117.csv`, with the 136-study coding sheet that preceded the confirmation pass in `reporting-audit-136.csv`. `scripts/aggregate_reporting_audit.py` regenerates the overall rates, the breadth counts, and the per-stage table from the 117-study file. Full texts were read for 110 studies; the other seven were coded from their abstracts and metadata. Per-study QA scores (QA1-QA4) for all 117 studies are in `data/reporting-compliance.json`.

## Screening Decisions

`data/screening-decisions/` publishes the decision records that can be released without exposing unarchived working files:

- `final-corpus-117.csv`: all 117 included studies with source, venue type, quality tier, RQ, categories, and rationale. `final-corpus-152.csv` keeps the pre-confirmation set.
- `window-2026-arxiv.csv`: 111 arXiv candidates for 2026-04-01..2026-08-31 with INCLUDE/EXCLUDE and reasons.
- `monthly-2026-10.csv`: 20 OpenAlex selections for 2026-09-01..2026-10-31, 11 included and 9 excluded with reasons.
- `coverage-audit-2026-10.csv`: the 400-record coverage-audit sample with both automated screeners, criteria, reasons, and adjudication.
- `gap-2026-03.csv`: March 2026 gap-search adjudication records.
- `original-2026-03/`: recovered March 2026 working records, including the 22,119-row Stage-1 auto-screening table, the 1,700-row scripted first pass and its automated re-screening, the recovered classifier and screening scripts, and the early PRISMA draft.
- `dual-screening-kit/`: blank 200-record title-and-abstract and 170-paper full-text dual-screening sheets, with the sampling frame and a kappa script. The sheets are instruments, not filled reliability tables.
- `stage1-classifier/`: a clean reconstruction of the published Stage-1 scoring rule; the recovered original is in `original-2026-03/scripts/03-screening/`.
- `RECONSTRUCTION.md`: what is public, what is missing, and how to close the gap.

The recovered March 2026 records are working data, not the final October 2026 117-study corpus. The final decisions are in `final-corpus-117.csv`. See `data/screening-decisions/original-2026-03/README.md` for provenance and caveats.

## Quality Tiers

`data/venues/venue-tier-117.csv` lists the venue type and quality tier of every study: CCF 2022 catalogue first (A/B/C), then CORE A* for venues outside the CCF catalogue (ICLR only in this corpus), workshops and unlisted venues as unranked, preprints as n/a. Distribution of the 45 peer-reviewed studies: 23 CCF-A, 7 CCF-B, 6 CCF-C, 1 CORE-A*, 8 unranked; the 72 preprints are n/a. The same column is in `primary-studies.csv`.

## RQ6: Empirical Composition and Pareto Experiments

Three controlled experiments evaluate efficiency techniques in combination, something primary-study classification alone cannot answer:

1. **Factorial Composition** (12 configurations on Qwen2.5-Coder-7B-Instruct): crosses quantization (FP16/INT8/INT4) x decoding (standard/speculative) x sampling (greedy/adaptive) and measures interaction effects on pass@1, latency, memory, throughput, and energy.
2. **Pareto Frontier** (15 configurations): 5 model scales (0.5B/1.5B/3B/7B/14B) x 3 precisions (FP16/INT8/INT4) under greedy decoding to map the efficiency-quality trade-off space.
3. **Energy Round** (27 configurations, single pass): NVML `nvmlDeviceGetTotalEnergyConsumption`-based energy measurement for every configuration, validated on V100 driver 570+.

Original campaign hardware: 4x NVIDIA Tesla V100-SXM2-32GB, PyTorch 2.5.1 + CUDA 12.4. All scripts and the design document are in `experiments/`.

The extended suite (2026) adds: five-run A800 replication of the composition and Pareto cells at six scales including 32B; EvalPlus re-judgment (HumanEval+) of every central cell on V100, A800, and a consumer RTX 3090 Ti third-architecture reference point; MBPP (five runs) and BigCodeBench (four runs, plus an instruct-protocol check on the lab V100-PCIE server); a contamination-free LiveCodeBench slice (288 post-release problems); pre-registered hypothesis tests H1-H5; a second model family (DeepSeek-Coder-6.7B-Instruct) at the four central cells; and instrumented draft-acceptance measurements on three architectures. Raw data, per-platform software stacks, quarantined defective outputs, the dropped MBPP+ layer, and the files lost when the second A800 rental expired are documented in `data/EXPERIMENT-DATA.md`.

## Data Description

### primary-studies.csv (117 rows)

| Column | Description |
|--------|-------------|
| ID | Sequential identifier from the 152-candidate set; seventeen identifiers are absent after the confirmation pass. S123-S141 are the 19 merged window-search studies, S142-S152 the 11 October additions |
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

- **Year distribution**: 2022 (1), 2023 (13), 2024 (28), 2025 (39), 2026 (36)
- **Venue types**: 33 conference, 12 journal, 72 preprint (62% preprints; 45 peer-reviewed)
- **Quality tiers** (peer-reviewed subset): 23 CCF-A, 7 CCF-B, 6 CCF-C, 1 CORE-A*, 8 unranked
- **RQ marginals** from the canonical category mapping: RQ1 24, RQ2 20, RQ3 59, RQ4 26, RQ5 15; 27 studies span two RQs

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
| **Seven-library total (main pass)** | **26,265** |
| **Update searches, 2026-04 to 2026-10** | **17,256** |
| **Total retrieved** | **43,521** |
| **After deduplication** | **39,374** |

Execution note: the Semantic Scholar bulk endpoint was queried twice on 2026-03-06; the working progress log records 864 hits from an earlier partial run, and 1,505 is the final protocol count reported above. The final execution record is with the first author; both counts are documented here and in `data/search-log.csv`.

Phase 4 (window search, publication window 2026-04-01 to 2026-08-31, run 2026-09): OpenAlex 436 candidates to 16 selected (`data/monthly-updates/2026-09.json`), arXiv 1,095 raw to 111 nominations to 6 selected (`data/window-search-2026/`), merged with 3 overlaps removed into 19 studies (`data/corpus-merge-20261001/`). A gap search covering 2026-03-07 to 2026-03-31 (run 2026-10-07) yielded 118 unique candidates, zero strictly eligible studies, and one record pending full-text adjudication (`data/gap-search-20261007/`). An independent OpenAlex coverage audit estimates campaign recall at 0.78 (95% CI 0.54-0.91) with three eligible records found in a 400-record sample (`data/coverage-audit-20261007/`).

## PRISMA Flow

```
43,521 records retrieved (cut-off 2026-10-31)
    26,265 from the seven-library pass of 2026-03-06
    17,256 from the update searches of April to October 2026
    |
    v  Deduplication (4,147 removed)
39,374 unique records
    |
    v  Title & abstract screening (35,534 excluded)
 3,840 candidates
    |
    v  Full-text retrieval: 1,730 of 3,840
    v  (2,110 candidates below the retrieval cut were never assessed)
    v  Full-text screening: scripted first pass, author check of every
    v  candidate (1,611 excluded)
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
117 primary studies (cut-off 2026-10-31)
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

The companion website (https://1719930244.github.io/efficient-codegen-slr-replication/) is updated on a monthly cadence: newly published studies are logged in `data/monthly-updates/` and rendered on the site without re-opening the review or changing the corpus. Process: [`docs/UPDATE-PROCESS.md`](docs/UPDATE-PROCESS.md); ledger: `data/monthly-updates/UPDATE-LOG.md`. The May and September 2026 tracker batches predate the corpus merge; the September batch's selected studies (plus the arXiv pass, minus overlaps) are the 19 studies merged into the 141-study window, and the October batch adds the final 11 studies.

## Known Gaps (recorded honestly)

1. **Stage-1 classifier artifacts**: recovered and mirrored in `data/screening-decisions/original-2026-03/`. The original `screening-ta-auto.py` is rule-based, not a weighted threshold; the recovered auto-screening table has 22,119 rows. Re-running it on that table labels 3,159 records Likely-Include or Uncertain; the final list of 3,810 candidates adds records retained by manual overrides and by the merge of export batches. See `RECONSTRUCTION.md`.
2. **Full-text screening decisions**: recovered and mirrored in `data/screening-decisions/original-2026-03/`. The scripted first pass labels 760 Include, 586 Uncertain, and 354 Exclude; an automated re-screening then labels 104 INCLUDE, 1,534 EXCLUDE, and 62 NO_TEXT. The authors checked every paper that the re-screening included, which gave the 89 full-text inclusions; see `original-2026-03/README.md`. Reliability was measured by dual-author validation on stratified samples of Stage 1 and Stage 3 (manuscript Section 3.3).
3. **Gap-search record G000** (EAAI 2026, DOI 10.1016/j.engappai.2026.114360): full text requires institutional access; its IC4 adjudication is pending.
4. Per-study 8-item audit: the coded sheets are in `data/reporting-audit/` (136 coded rows, 117 of them in the final corpus); `scripts/aggregate_reporting_audit.py` regenerates every rate.

## License

This replication package is released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
