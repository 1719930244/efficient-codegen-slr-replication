# Changelog

## v2.0 (2026-10-09) — revision release, 141-study corpus

Synchronizes the package with the revised manuscript (TOSEM-2026-0589 R1).

- Corpus: `data/primary-studies.csv` 122 -> 141 rows (19 window-search studies merged, Phase 4, cut-off 2026-08-31). Merge is fully reproducible in-package: `data/corpus-merge-20261001/recompute.py` regenerates the merged CSV byte-identically from the archived 122-era inputs plus the per-study extraction records (`audit-A.json`, `audit-B.json`).
- `RQ` column regenerated from the canonical stage mapping of `scripts/plot_upset.py` (replaces the incomplete legacy annotation); marginals 29/26/63/28/27, 32 multi-stage studies.
- New `Quality Tier` column and `data/venues/venue-tier-141.csv` (CCF 2022 / CORE A* / unranked rules; peer-reviewed distribution 27/10/6/1/11).
- `data/reporting-compliance.json` extended to 141 studies; `data/statistics.json` regenerated (141, five-RQ marginals, per-category counts from primary categories).
- `data/by-rq/` regenerated for all five RQs with the full column schema (rq5 file added).
- `data/classification-scheme.csv` study counts and keys refreshed for 141.
- New: `data/search-log.csv` and `docs/search-queries.md` (two-group query protocol, per-database queries verbatim, window/gap/audit records, Semantic Scholar dual-run note).
- New artifact directories: `data/corpus-merge-20261001/`, `data/window-search-2026/` (arXiv pass with per-record screening log), `data/gap-search-20261007/`, `data/coverage-audit-20261007/` (recall 0.78, 95% CI 0.54-0.91).
- New scripts: `scripts/sensitivity_findings.py` (reproduces Section 3.8 Tables 3-4 on the 141 CSV), `scripts/plot_pareto_a800_cleanmem.py` (clean-memory Pareto render), `scripts/monthly_update_openalex.py` (standard monthly living-update search); `scripts/plot_upset.py` updated for the 141 corpus; all plotting/analysis scripts now default to package-relative paths.
- `generate-data.py` marked deprecated (its screening-worksheet inputs were never archived).
- Companion website synchronized to the 141 corpus (six RQs, eight findings, RQ6 experiments section, monthly update cadence).
- README rewritten; known gaps recorded honestly (classifier artifacts, 1,700 full-text decisions, G000 adjudication, 122-era audit sheets).

## v1.3 (2026-09-22)

- September round addendum: three-run energy-variance replicate of the FP16 LiveCodeBench cell (`data/v100-sept-round-216/lcb-energy-variance`).

## v1.2 (2026-09-21)

- September 2026 V100/3090Ti replication round (`data/v100-sept-round-216`, 78 files).
- Venue Type annotation completed for all 122 studies (16 blanks filled mechanically from Venue).
- Lab server SKU corrected to V100-PCIE-32GB in the LiveCodeBench layer docs.

## v1.1 (2026-09-10)

- A800 extension raw data (both rental servers, final states before retirement).
- RTX 3090 Ti third-architecture reference point with judged artifacts.
- V100 original-campaign April logs and dependency manifests; refreshed patched script suite.
- LiveCodeBench contamination-free layer (288 post-release problems, official judge).
- EXPERIMENT-DATA.md provenance map (per-platform stacks, quarantine manifest, dropped MBPP+ layer, rental-expiry losses).

## v1.0 (2026-05-08)

- Initial release accompanying the first submission: 122-study corpus, classification scheme, statistics, RQ6 experiment code, companion website.
