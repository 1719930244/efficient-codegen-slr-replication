# Reporting audit (8 items)

Per-study coding of how each primary study reports its experimental setup and
efficiency. This folder holds the audit behind Table 3, Table 4, and Figure 6
of the manuscript.

## Files

| File | Rows | Use |
|------|------|-----|
| `reporting-audit-136.csv` | 136 | All studies coded in the audit, including the 19 that left the corpus at the confirmation pass. |
| `reporting-audit-117.csv` | 117 | The final corpus. **This is the file the manuscript's rates use.** |

## Coding rules

Read the abstract, the experimental setup, the results, and the conclusion, and
code `YES` only when the paper states the item with a concrete value or
specification, `NO` when it does not, and `UNCLEAR` only for genuine ambiguity.

1. `Correctness`: reports pass@k, functional correctness, or an equivalent task-success number.
2. `ModelName`: names the model and its version or size.
3. `Hardware`: states the GPU or CPU used for the experiments.
4. `Latency`: reports generation-side latency or throughput. The runtime of the generated program does not count.
5. `Memory`: reports model or inference memory. Memory used by the generated program does not count.
6. `ServingConfig`: reports batch size, concurrency, serving framework, decoding configuration, or an equivalent deployment setting.
7. `MonetaryCost`: reports dollar or API cost.
8. `Energy`: reports model or inference energy. Energy used by the generated program does not count.

Additional columns: `NEfficiencyMetrics` counts the distinct families reported
with a number, `BothLatencyMemory` and `BothLatencyCost` mark co-reporting, and
`Evidence` quotes the sentence on which a `YES` rests.

## Denominators

Full texts were read for 110 of the 117 studies. The other seven were coded
from their abstracts and metadata and carry `FullTextRead=NO`. The manuscript
reports all rates over the 117 studies, that is, over the whole corpus, so its
rates and the corpus-level tables use one denominator.

## Rates over the 117 studies

| Item | Studies | Rate |
|------|---------|------|
| Correctness | 103 | 88% |
| Model name / version | 107 | 91% |
| Hardware | 77 | 66% |
| Latency / throughput | 58 | 50% |
| Memory | 19 | 16% |
| Serving configuration | 80 | 68% |
| Monetary cost | 19 | 16% |
| Energy | 8 | 7% |

Reporting breadth: 20 studies report no efficiency metric, 39 report one, 31
report two, 22 report three, and 5 report four. Twelve studies report latency
and memory together.

## Stage breakdown

Stages follow the canonical category mapping used for the RQ marginals:
RQ1 24, RQ2 20, RQ3 59, RQ4 26, RQ5 15, with 27 studies counted in two stages.

| Stage | N | Latency | Memory | Energy | Monetary cost |
|-------|---|---------|--------|--------|---------------|
| Data (RQ1) | 24 | 29% | 17% | 0% | 17% |
| Training (RQ2) | 20 | 40% | 20% | 0% | 5% |
| Inference (RQ3) | 59 | 58% | 10% | 7% | 12% |
| Deployment (RQ4) | 26 | 65% | 31% | 12% | 31% |
| Evaluation (RQ5) | 15 | 27% | 33% | 27% | 7% |

## Provenance

The coding was done in seventeen batches against the full texts collected for
the 152-study confirmation pass. The per-batch inputs are not redistributed
because they contain local file paths; the coded outputs are complete and are
what the aggregate figures are computed from. Aggregate helper:
`scripts/aggregate_reporting_audit.py` regenerates every rate above from
`reporting-audit-117.csv`.