# Stage-1 classifier reconstruction

This directory contains a transparent reconstruction of the Stage-1 keyword classifier described in Section 3.3 of the manuscript.

`classify.py` implements the published scoring rule:

- match the Group A, B, and C vocabularies against title and abstract;
- compute one weighted score per group;
- set the overall score to zero when any group is unmatched;
- retain records whose score reaches the threshold;
- rank retained records by score.

## Important limitation

The **executed threshold and per-group weights are not stored in this package**, and the **22,118-record input list is not mirrored here**. The original classifier artifacts were kept outside the released package. This script is therefore a reconstruction of the rule, not the original code, and it cannot reproduce the published 3,810 retained candidates without the original threshold, weights, and input records.

Do not infer the executed values from this directory. Pass them explicitly:

```bash
python classify.py --input records.jsonl --output ranked.csv \
  --threshold <executed-threshold> --weights <A>,<B>,<C>
```

The package publishes the final decisions and the post-screening batches under `data/screening-decisions/`. The per-paper decisions for the 1,700 full texts are not mirrored; see `../RECONSTRUCTION.md`.