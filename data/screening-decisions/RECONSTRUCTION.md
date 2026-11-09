# Reconstruction status of the missing screening artifacts

The package publishes every screening decision that can be reproduced from the archived files. Two artifact families are not mirrored:

1. **Stage-1 classifier artifacts**: the executed code, threshold, per-group weights, and the ranked list of 22,118 unique records.
2. **1,700 full-text decisions**: the per-paper INCLUDE/EXCLUDE records with criterion codes, or the original worksheets.

## What is public

- `final-corpus-152.csv`: final included studies with source, venue, quality tier, RQ, categories, and rationale.
- `window-2026-arxiv.csv`: 111 window-search candidates with decisions and reasons.
- `monthly-2026-10.csv`: 20 October selections, 11 included and 9 excluded with reasons.
- `coverage-audit-2026-10.csv`: 400-record coverage audit with both screeners and adjudication.
- `gap-2026-03.csv`: 38 March gap-search adjudications.

## What is not public

- `stage1-classifier/classify.py` is a reconstruction of the published rule, not the executed code. The executed threshold and weights are unavailable.
- The 22,118-record input and the 1,700-paper full-text worksheets are unavailable in this package and in the authors' local backups. They are not reconstructed or invented here.

## How to close the gap

If the original classifier configuration and the 1,700-record input list are recovered, place them here and run:

```bash
python stage1-classifier/classify.py --input <records> --output ranked.csv \
  --threshold <executed> --weights <A>,<B>,<C>
```

If the full-text worksheets are recovered, use the columns in `fulltext-decision-template.csv` and merge them into `final-corpus-152.csv` by `id`/`key`.

## Where the originals were

The project's own documents record the original locations:

- `data/coverage-audit-20261007/REPORT.md` states that the Stage-1 records (22,118), the 3,810 ranked list, the classifier code, and the 1,700 full-text decisions were kept at `/home/szw/overleaf/efficient-codegen-survey/data/` on the A800 rental server, which has been retired.
- `data/ITX-handoff-20261007.md` adds that the originals may also be on the office machine at `/home/szw/school/efficient-codegen-survey/` or in the ITX screening working directory. It also notes that `data/screening-ta-auto.csv` is an early 3,367-record version, not the final 22,118-record list.
- The protocol documents name the target files: `screening-ta.csv`, `screening-ft.csv`, `entries-summary.csv`, `likely-include.bib`, and `prisma-counts.json`.

## Servers checked on 2026-11-09

- 215: all readable home directories checked; no Stage-1 or full-text decision files.
- 217: readable home directories checked; no Stage-1 or full-text decision files.
- 216: reachable only as `liyuning`; `/home/v3090ti` is not readable, so `/home/v3090ti/szw_alibaba_backup/04_efficient-codegen-survey` could not be checked.
- A800-1 / A800-2: hostnames no longer resolve; retired.
- sg: reachable; `/root/szw/efficient-codegen-survey`, `/home/szw/tosem-survey-hub`, and `/root/szw/tosem-r1-work` checked; the classifier and 1,700-paper worksheets are not there.

The evidence documents are archived with the submission package under `evidence/`.
