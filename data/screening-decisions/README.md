# Public screening decisions

This directory publishes the screening decisions that can be released without exposing unarchived working files. It covers the final corpus, the April-August 2026 window pass, the September-October 2026 monthly batch, the coverage audit sample, and the March 2026 gap search.

| File | Contents |
|---|---|
| `final-corpus-116.csv` | All 116 included studies with source, venue type, quality tier, RQ, categories, and rationale. |
| `window-2026-arxiv.csv` | 111 arXiv candidates for 2026-04-01..2026-08-31 with INCLUDE/EXCLUDE and reason. |
| `monthly-2026-10.csv` | 20 OpenAlex selections for the 2026-09-01..2026-10-31 window: 11 included, 9 excluded with reasons. |
| `coverage-audit-2026-10.csv` | 400-record coverage-audit sample with both automated screeners, criteria, reasons, and adjudication. |
| `gap-2026-03.csv` | March 2026 gap-search adjudication records. |

## What is not in this directory

The Stage-1 classifier code and configuration, the ranked list of 39,374 unique records, and the per-paper INCLUDE/EXCLUDE decisions for the 1,700 full texts of the main pass are not mirrored here. The classifier artifacts and the 1,700-paper decision worksheets were kept outside the package; they are available from the authors. The package does publish every decision needed to reconstruct the final 116-study corpus from the released search log and the files above.

## Counts

- Final corpus: 116 included studies.
- Window search: 111 candidates, 19 included after screening.
- Monthly October batch: 20 selected, 11 included, 9 excluded.
- Coverage audit: 400 sampled records, two screeners, adjudication recorded.
