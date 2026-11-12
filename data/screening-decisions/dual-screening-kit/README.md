# Dual-screening kit: 200 title-and-abstract records, 170 full texts

This directory contains the two blank dual-screening instruments described in Section 3.3 and the response to Reviewer 2 Cons #2.

## Title-and-abstract kit

- `TA-screening-Rater1.xlsx`, `TA-screening-Rater2.xlsx`: 200 records drawn from the recovered Stage-1 auto-screening table. The sample is 100 retained records (42 Likely-Include and 58 Uncertain) and 100 Likely-Exclude records, with seed 20261007. The decision, criterion, and note columns are blank.
- `sample-ids-ta.json`: the 200 sampled IDs and strata.
- `build_kit_ta.py`: the script that built the two sheets.

Abstracts are present for 89 of the 200 records. The remaining records were title-only in the Stage-1 input, so their abstract cells are empty by design. The sample is a reconstruction of the Stage-1 sampling frame, not the earlier coverage-audit sample.

## Full-text kit

- `FT-screening-Rater1.xlsx`, `FT-screening-Rater2.xlsx`: 170 full texts, a 10% stratified sample of the 1,700 retrieved papers. The sample is 76 Include, 59 Uncertain, and 35 Exclude from the recovered first-pass worksheet.
- `sample-ids-ft.json`: the 170 sampled IDs and strata.
- The decision, criterion, and note columns are blank.

The 170-row sample was drawn from `../original-2026-03/03-screening/screening-ft.csv` with `random.Random(20261007)`, stratified by the first-pass decision. It is a reconstruction of the sampling frame, not a record of a completed second screening.

## Computing kappa

After both raters fill the decision columns, run:

```bash
python kappa.py TA-screening-Rater1.xlsx TA-screening-Rater2.xlsx
python kappa.py FT-screening-Rater1.xlsx FT-screening-Rater2.xlsx
```

The script reports three-class agreement and kappa, binary agreement and kappa with Uncertain mapped to Include, and the disagreement IDs.

## Important limitation

The blank sheets do not reproduce the kappa values reported in the manuscript. Those values require the filled second-rater decisions. The recovered March 2026 records contain the first-pass full-text decisions and the 200-record T&A sample, but not a filled second-rater sheet. The kit is published so the dual-screening procedure can be executed and checked; it is not evidence that the reported kappa was computed from these files.