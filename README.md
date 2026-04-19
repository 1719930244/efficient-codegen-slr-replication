# Replication Package: Towards Efficient LLM-Based Code Generation

Replication package for the systematic literature review: *"Towards Efficient LLM-Based Code Generation: A Systematic Review"* (submitted to ACM TOSEM).

## Overview

This review systematically analyzes **122 primary studies** on efficiency techniques for LLM-based code generation, organized across **six research questions** spanning the full lifecycle from data preparation through deployment, plus a cross-cutting empirical study on technique composition.

| RQ | Scope | Studies |
|----|-------|---------|
| RQ1 | Data Preparation (selection, quality, synthesis) | 25 |
| RQ2 | Model Training (pre-training, PEFT, distillation, RL, curriculum) | 24 |
| RQ3 | Inference Optimization (decoding, input/output compression, serving, routing, post-training compression, code-specific) | 59 |
| RQ4 | Deployment (deployment-stage optimization) | 24 |
| RQ5 | Evaluation (benchmarks, metrics, reporting compliance) | 21 |
| RQ6 | Technique Composition and Interaction Effects (empirical experiments) | — |

> Note: 31 studies span multiple RQs (per-RQ counts sum to more than 122). RQ6 is answered by controlled experiments rather than primary-study classification (see `experiments/`).

## Repository Structure

```
data/
  primary-studies.csv          # All 122 primary studies with metadata and classification
  classification-scheme.csv    # Taxonomy: 24 categories mapped to study keys
  statistics.json              # Summary statistics (year, venue, RQ, category distributions)
  reporting-compliance.json    # Per-study reporting compliance (8-item audit, N=122)
  by-rq/
    rq1-studies.csv            # RQ1: Data Preparation studies
    rq2-studies.csv            # RQ2: Model Training studies
    rq3-studies.csv            # RQ3: Inference Optimization studies
    rq4-studies.csv            # RQ4: Deployment studies
    rq5-studies.csv            # RQ5: Evaluation studies
experiments/                   # RQ6 empirical experiments
  EXPERIMENT-PLAN.md           # Design doc: factorial composition + Pareto frontier + energy round
  scripts/
    download_models.py
    eval_humaneval.py          # HumanEval evaluation core (NVML energy measurement included)
    run_composition.py         # Experiment 1: 12 configs (3 precision x 2 decoding x 2 sampling)
    run_pareto.py              # Experiment 2: 15 configs (5 model scales x 3 precisions)
    run_energy_round.py        # Experiment 3: single-pass energy measurement for 27 configs
    launch-pareto-parallel.sh  # GPU watcher: auto-schedule pareto run2/run3
    launch-energy-round.sh     # GPU watcher: auto-schedule energy round on freed GPU
    plot_results.py            # Figure generation (Pareto 4-subplot)
scripts/
  generate-data.py             # Script to regenerate data tables from source files
TAXONOMY.md                    # Full classification taxonomy with descriptions
```

## Classification Taxonomy

The taxonomy follows the LLM lifecycle with fine-grained categories organized across RQ1-RQ5. A single primary study may map to multiple categories.

### RQ1: Data Preparation (25 studies)
| Code | Category |
|------|----------|
| 1a | Data Selection |
| 1b | Data Quality Assessment & Synthesis |

### RQ2: Model Training (24 studies)
| Code | Category |
|------|----------|
| 2a | Efficient Pre-training |
| 2b | Parameter-Efficient Fine-Tuning (PEFT) |
| 2c | Knowledge Distillation |
| 2d | Curriculum Learning |
| 2f | RL-based Training |

### RQ3: Inference Optimization (59 studies)
| Code | Category |
|------|----------|
| 3a | Speculative Decoding |
| 3b | Early Exit |
| 3c | Non-Autoregressive Generation |
| 3d | Prompt Compression |
| 3e | Context Pruning |
| 3f | KV Cache Optimization |
| 3g | Post-Training Quantization |
| 3h | Model Pruning |
| 3j | Adaptive Sampling |
| 3l | Code-Specific Optimization |
| 3m | System-Level Serving |
| 3n | Prompt Engineering |
| 3o | Chain-of-Thought Optimization |

### RQ4: Deployment (24 studies)
| Code | Category |
|------|----------|
| 3i | Model Routing |
| 3k | Multi-Agent Orchestration |

> Deployment-stage optimizations (routing and multi-agent orchestration) were separated from RQ3 to reflect their operational rather than model-internal nature.

### RQ5: Evaluation (21 studies)
| Code | Category |
|------|----------|
| 4a | Benchmark Design |
| 4b | Empirical Study |

See `TAXONOMY.md` for full descriptions and `data/classification-scheme.csv` for study-to-category assignments.

## Reporting Compliance Audit (N=122)

All 122 primary studies were audited against an 8-item reporting checklist. Aggregate coverage:

| Item | Coverage |
|------|----------|
| Functional Correctness | 96% |
| Model Info | 97% |
| Hardware | 69% |
| Latency / Throughput | 48% |
| Memory | 19% |
| Serving Config | 39% |
| Monetary Cost | 20% |
| Energy / Carbon | 7% |

Depth breakdown: 26% of studies report no efficiency metric, 34% report exactly one, and 40% report two or more. See `data/reporting-compliance.json` for per-study scores.

## RQ6: Empirical Composition and Pareto Experiments

Three controlled experiments evaluate efficiency techniques in combination, something primary-study classification alone cannot answer:

1. **Factorial Composition** (12 configurations on Qwen2.5-Coder-7B-Instruct): crosses quantization (FP16/INT8/INT4) x decoding (standard/speculative) x sampling (greedy/adaptive) and measures interaction effects on pass@1, latency, memory, throughput, and energy.
2. **Pareto Frontier** (15 configurations): 5 model scales (0.5B/1.5B/3B/7B/14B) x 3 precisions (FP16/INT8/INT4) under greedy decoding to map the efficiency-quality trade-off space.
3. **Energy Round** (27 configurations, single pass): NVML `nvmlDeviceGetTotalEnergyConsumption`-based energy measurement for every configuration, validated on V100 driver 570+.

Hardware: 4x NVIDIA Tesla V100-SXM2-32GB, PyTorch 2.5.1 + CUDA 12.4. All scripts and the design document are in `experiments/`.

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

- **Year distribution**: 2020 (1), 2022 (1), 2023 (15), 2024 (35), 2025 (54), 2026 (19)
- **Sources**: Database search + snowballing + supplementary search

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
