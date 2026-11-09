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