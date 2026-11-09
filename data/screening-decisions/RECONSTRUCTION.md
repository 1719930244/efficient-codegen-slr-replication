# Screening records: recovered files and remaining gaps

The Stage-1 auto-screening table, the classifier and full-text screening scripts, and the 1,700-paper first-pass worksheet were recovered on 2026-11-09 from the 11.7 GB backup `04_efficient-codegen-survey.zip` on server 216. They are mirrored in `original-2026-03/`.

## What is public

- `original-2026-03/screening-ta-auto.csv`: 22,119 rows, Stage-1 automatic labels (1,380 Likely-Include, 1,779 Uncertain, 18,960 Likely-Exclude; 22,114 unique keys).
- `original-2026-03/screening-ft.csv` and `screening-ft-results.xlsx`: 1,700 rows, first-pass full-text decisions (760 Include, 586 Uncertain, 354 Exclude).
- `original-2026-03/screening-llm-merged.csv` and `screening-llm-results.xlsx`: 1,700 rows, LLM-assisted full-text pass (104 INCLUDE, 1,534 EXCLUDE, 62 NO_TEXT).
- `original-2026-03/entries-summary.csv`: 22,119 deduplicated records with title, author, year, DOI, arXiv ID, source, and type.
- `original-2026-03/likely-include.bib`: BibTeX for the likely-include set.
- `original-2026-03/fine-grained-classification.csv`: 125-row classification from the 122-era corpus.
- `original-2026-03/05-prisma/prisma-counts.json`: early PRISMA draft (4,179 raw, 369 duplicates, 3,810 after dedup).
- `original-2026-03/scripts/03-screening/`: the recovered `screening-ta-auto.py`, `screening-ft.py`, `screening-ft-llm.py`, and helper scripts.
- `final-corpus-152.csv`: the final included corpus.
- `window-2026-arxiv.csv`, `monthly-2026-10.csv`, `coverage-audit-2026-10.csv`, `gap-2026-03.csv`: the post-screening batches.
- `stage1-classifier/classify.py`: a clean reconstruction of the published rule set, kept as a readable companion to the recovered script.

## What remains different from the final manuscript

- The recovered auto-screening table is the March 2026 working export. It has 22,119 rows and 22,114 unique keys; the manuscript's final PRISMA uses 22,118 deduplicated records. The difference is an export artifact.
- The recovered table labels 3,159 records as Likely-Include or Uncertain. The final pipeline retained 3,810 candidates after later manual overrides and export merging. The recovered table is the closest surviving source, not a byte-level reproduction of the final 3,810-row list.
- The 1,700-paper worksheet is the first-pass and LLM-assisted working record. The final 89/122/141/152-study corpus went through later manual review, snowballing, targeted search, and the April-October 2026 window updates. The 760 Include rows are not the final included set.
- `fine-grained-classification.csv` has 125 rows and belongs to the 122-era corpus; the final 152-study classification is in `final-corpus-152.csv`.
- `prisma-counts.json` is an early draft (4,179 raw, 3,810 after dedup), not the final 26,265/22,118/3,810 PRISMA in the manuscript.

## Provenance

- Source archive: `/home/v3090ti/szw_alibaba_backup/04_efficient-codegen-survey.zip` (11.7 GB) on 216.
- Internal path: `home/szw/school/efficient-codegen-survey/data/`.
- Recovered: 2026-11-09, read-only extraction to `/tmp/tosem-artifacts/` on 216, then copied to this package.
- The archive remains on 216; nothing was deleted or modified there.

## How to use the recovered records

The recovered scripts expect the office-machine data layout (`data/deduplicated.bib`, `data/03-screening/`, and so on). The extracted package mirrors the relevant subdirectories. To rerun the Stage-1 classifier, point `DATA_DIR` in `screening-ta-auto.py` at the package directory containing `deduplicated.bib` and run it. To rerun the full-text pass, use `screening-ft.py`; it expects the first-pass worksheet and the paper metadata in the same layout.

The clean reconstruction in `stage1-classifier/classify.py` is a simplified, dependency-free implementation of the same rule set for readers who want to inspect the logic without the original working directory.