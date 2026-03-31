# Replication Package: Efficiency in LLM-based Code Generation

Replication package for the systematic literature review: *"Efficiency in LLM-based Code Generation: A Systematic Literature Review"*.

## Overview

This review systematically analyzes **122 primary studies** on efficiency techniques for LLM-based code generation, organized across four research questions spanning the full lifecycle from data preparation to evaluation.

| RQ | Scope | Studies |
|----|-------|---------|
| RQ1 | Data Preparation (selection, quality, ordering) | 10 |
| RQ2 | Model Training (pre-training, PEFT, distillation, RL) | 29 |
| RQ3 | Inference Optimization (decoding, compression, routing, serving) | 87 |
| RQ4 | Evaluation (benchmarks, metrics, empirical studies) | 21 |

> Note: 25 studies span multiple RQs, so per-RQ counts sum to more than 122.

## Repository Structure

```
data/
  primary-studies.csv          # All 122 primary studies with metadata and classification
  classification-scheme.csv    # Taxonomy: 24 categories mapped to study keys
  statistics.json              # Summary statistics (year, venue, RQ, category distributions)
  reporting-compliance.json    # Per-study reporting compliance (QA scores)
  by-rq/
    rq1-studies.csv            # RQ1: Data Preparation studies
    rq2-studies.csv            # RQ2: Model Training studies
    rq3-studies.csv            # RQ3: Inference Optimization studies
    rq4-studies.csv            # RQ4: Evaluation studies
scripts/
  generate-data.py             # Script to regenerate data from source files
TAXONOMY.md                    # Full classification taxonomy with descriptions
```

## Classification Taxonomy

The taxonomy is organized by lifecycle stage with 24 fine-grained categories:

### RQ1: Data Preparation
| Code | Category | Count |
|------|----------|-------|
| 1a | Data Selection | 9 |
| 1b | Data Quality Assessment & Synthesis | 4 |

### RQ2: Model Training
| Code | Category | Count |
|------|----------|-------|
| 2a | Efficient Pre-training | 3 |
| 2b | Parameter-Efficient Fine-Tuning (PEFT) | 10 |
| 2c | Knowledge Distillation | 6 |
| 2d | Curriculum Learning | 2 |
| 2f | RL-based Training | 10 |

### RQ3: Inference Optimization
| Code | Category | Count |
|------|----------|-------|
| 3a | Speculative Decoding | 3 |
| 3b | Early Exit | 5 |
| 3c | Non-Autoregressive Generation | 3 |
| 3d | Prompt Compression | 7 |
| 3e | Context Pruning | 8 |
| 3f | KV Cache Optimization | 3 |
| 3g | Post-Training Quantization | 7 |
| 3h | Model Pruning | 2 |
| 3i | Model Routing | 11 |
| 3j | Adaptive Sampling | 8 |
| 3k | Multi-Agent Orchestration | 11 |
| 3l | Code-Specific Optimization | 9 |
| 3m | System-Level Serving | 5 |
| 3n | Prompt Engineering | 8 |
| 3o | Chain-of-Thought Optimization | 5 |

### RQ4: Evaluation
| Code | Category | Count |
|------|----------|-------|
| 4a | Benchmark Design | 3 |
| 4b | Empirical Study | 18 |

## Data Description

### primary-studies.csv

| Column | Description |
|--------|-------------|
| ID | Sequential identifier (S001-S122) |
| Key | Citation key used in the paper |
| Title | Full paper title |
| Year | Publication year (2020-2026) |
| Venue | Publication venue |
| RQ | Research question assignment (may be multiple, separated by `;`) |
| Primary Categories | Fine-grained classification codes (separated by `;`) |
| Secondary Categories | Additional classification codes, if any |
| Scope Flags | Scope annotations, if any |
| Brief Rationale | Classification rationale |

### Study Characteristics

- **Year distribution**: 2020 (1), 2022 (1), 2023 (14), 2024 (35), 2025 (52), 2026 (19)
- **Sources**: Database search, snowballing, and supplementary search

## Search Strategy

Seven digital libraries were searched using a structured query combining three concept groups:

- **Group A** (Code Generation): code generation, code completion, program synthesis, ...
- **Group B** (Efficiency): efficiency, optimization, latency, throughput, ...
- **Group C** (LLM/Transformer): large language model, transformer, neural network, ...

Query: `(Group A) AND (Group B) AND (Group C)`, year >= 2017

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

## PRISMA Flow

```
26,265 raw records (7 databases)
    |
    v  Deduplication
22,118 unique records
    |
    v  Title & Abstract screening
 3,810 candidates (18,308 excluded)
    |
    v  Full-text retrieval + screening
    89 included (1,611 excluded after full-text review)
    |
    +-- Snowballing: +18 (2,645 candidates, 2,627 excluded)
    +-- Supplementary search: +18
    |
    v  Quality assessment (4 criteria x 2 reviewers)
  125 candidates
    |
    v  Duplicate removal (-3)
  122 primary studies
```

## Quality Assessment

Each study was assessed on four criteria by two independent reviewers:

| Criterion | Description |
|-----------|-------------|
| QA1 | Does the study include a quantitative empirical evaluation? |
| QA2 | Does the study report at least one baseline comparison? |
| QA3 | Does the study disclose the experimental configuration? |
| QA4 | Does the study provide reproducibility artifacts? |

Scores: 1.0 (fully met), 0.5 (partially met), 0.0 (not met). Maximum total: 4.0.

## License

This replication package is released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
