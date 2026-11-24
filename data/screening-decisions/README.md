# Screening decisions

Decision records for every screening step of Section 3.3, from Stage 1 to the
confirmation pass, and for the coverage audit.

| Path | Contents |
|---|---|
| `stage1-classifier/` | Rule-based Stage-1 classifier, the Stage-1 labels of the main search with their matched-term counts, and a script that re-derives every label from the rule. See its README. |
| `fulltext-screening/first-pass.csv` | Scripted first pass over the full texts of the main search: decision, reason, and per-criterion scores (IC1-IC4, EC1-EC4). |
| `fulltext-screening/re-screening.csv` | Automated re-screening of the same full texts: decision and per-criterion flags. The authors checked every paper that the re-screening retained. |
| `window-2026-arxiv.csv` | The 111 arXiv candidates of the April-August 2026 update search, with decision and reason. |
| `monthly-2026-10.csv` | The 20 OpenAlex selections of the September-October 2026 update search: 11 included, 9 excluded with reasons. |
| `final-corpus-116.csv` | The 116 primary studies with source, venue type, quality tier, RQ, categories, and rationale. |
| `confirmation-exclusions-35.csv` | The 35 studies removed in the confirmation pass, with the failed criterion, the evidence, and the reason. |
| `coverage-audit/` | The coverage audit: 400-record sample with both screeners' decisions and criteria and the adjudication (`coverage-audit-400.csv`, `adjudication.json`), the OpenAlex query (`query-meta.json`), and the recall estimate (`estimate.py`, `estimates.json`). |

## Counts

- Full-text screening gave 119 inclusions: 89 from the full texts of the main
  search, and 30 from the update searches (19 from April-August, 11 from
  September-October).
- Backward and forward snowballing added 18 studies and the targeted search
  added 18; after 3 duplicates were removed, 152 candidates entered the
  confirmation pass.
- The confirmation pass removed 35 studies (`confirmation-exclusions-35.csv`),
  leaving 117; one study appears both as an ASE paper and as its arXiv version
  and was merged under EC5, which leaves the 116 studies of
  `final-corpus-116.csv`.
- Coverage audit: 400 sampled records, two screeners, adjudication recorded;
  `estimate.py` reproduces the recall of 0.78 (95% CI 0.54-0.91).
