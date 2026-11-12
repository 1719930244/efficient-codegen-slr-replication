# Recovered screening records (March 2026)

These files were recovered on 2026-11-09 from `04_efficient-codegen-survey.zip` in `/home/v3090ti/szw_alibaba_backup/` on server 216. The archive is a backup of the office machine directory `home/szw/school/efficient-codegen-survey/` and is dated 2026-03-10. The server-side copy is read-only; these files are mirrored here with their original names.

## Files

| File | Rows | Contents |
|---|---:|---|
| `screening-ta-auto.csv` | 22,119 | Stage-1 automatic title-and-abstract labels. Labels: 1,380 Likely-Include, 1,779 Uncertain, 18,960 Likely-Exclude, so 3,159 records are Likely-Include or Uncertain. 22,114 unique keys. |
| `screening-ft.csv` | 1,700 | Scripted first pass over the full texts. 760 Include, 586 Uncertain, 354 Exclude, with criterion and reason columns. |
| `screening-ft-results.xlsx` | 1,700 | Workbook form of the scripted first pass, with Primary Studies / Uncertain / Excluded / Summary sheets. |
| `screening-llm-merged.csv` | 1,700 | LLM re-screening of the full texts. 104 INCLUDE, 1,534 EXCLUDE, 62 NO_TEXT. |
| `screening-llm-results.xlsx` | 1,700 | Workbook form of the LLM re-screening, with Primary Studies / All Results / Summary sheets. |
| `entries-summary.csv` | 22,119 | Deduplicated record table with title, author, year, DOI, arXiv ID, source database, and type. |
| `likely-include.bib` | 2.1 MB | BibTeX records for the likely-include set. |
| `fine-grained-classification.csv` | 125 | Earlier fine-grained classification of the included set (122-era plus three). This is not the final 152-study corpus. |
| `prisma-counts.json` | 1 | Early PRISMA draft: 4,179 raw records, 369 duplicates removed, 3,810 after deduplication. This is not the final 26,265/22,118/3,810 PRISMA. |
| `screening-batches/` | 1,700 | Batched full-text screening files and progress JSON. |
| `scripts/03-screening/` | 5 scripts | `screening-ta-auto.py`, `screening-ft.py`, `screening-ft-llm.py`, `fetch-abstracts.py`, `retry-503.py`. |
| `scripts/06-classify/` | 3 scripts | `classify-papers.py`, `apply-classification.py`, `rebuild-assessment.py`. |
| `scripts/07-compliance/` | 27 files | Compliance audit batches and results. |

## Important caveats

- These are **March 2026 working records**, not the final October 2026 152-study corpus. The final corpus is `../final-corpus-152.csv`.
- The Stage-1 classifier in `scripts/03-screening/screening-ta-auto.py` is **rule-based**, not a weighted score with a fixed threshold. It counts matches in three vocabularies and applies the rules in `classify_entry`. Earlier drafts of the manuscript described a threshold/weight classifier; that description is superseded by this code.
- The 1,700 full-text rows record the scripted first pass and the LLM re-screening of Stage 3. The authors checked every paper that the re-screening included, which gave the 89 full-text inclusions; snowballing, targeted search, and the April-October 2026 window updates added the rest. Do not treat the 760 Include rows of the first pass as the final included set.
- `prisma-counts.json` is an early draft and does not match the final PRISMA in the manuscript.
- The recovered `screening-ta-auto.csv` has 22,119 rows and 22,114 unique keys; the manuscript's final PRISMA uses 22,118 deduplicated records. The difference is an export artifact, not a new screening result.

## Provenance

- Source archive: `/home/v3090ti/szw_alibaba_backup/04_efficient-codegen-survey.zip` (11.7 GB) on 216.
- Internal path: `home/szw/school/efficient-codegen-survey/data/`.
- Recovered: 2026-11-09, read-only extraction to `/tmp/tosem-artifacts/` on 216, then copied to this package.
- The archive remains on 216; nothing was deleted or modified there.