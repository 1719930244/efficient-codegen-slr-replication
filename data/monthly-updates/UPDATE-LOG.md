# Monthly Update Log

One row per batch or attempt. Funnel stages: Group A hits / Boolean pass /
title-strict nominated / selected. See docs/UPDATE-PROCESS.md for the SOP.

| Date | Window | Funnel (A/bool/nom/sel) | Site rendered | Commit | Pages verified | Notes |
|------|--------|-------------------------|---------------|--------|----------------|-------|
| 2026-05-28 | 2026-04-01..2026-05-28 | 610/221/86/16 (S2 bulk, title-strict) | yes | 0e6de43 | yes (at the time) | First batch. Funnel recorded in commit message; schema predates 2026-09. |
| 2026-09-07 | 2026-04-01..2026-09-07 | 436/-/23/16 (OpenAlex, ad hoc script not committed) | **rendered 2026-09-29** | 4ec4e2c (data), 0039c76 (site) | yes 2026-09-29 | Site card was missed on 09-07 and rendered on 09-29 with the May-overlap study marked. 1 study overlaps the May batch (shared window start). |
| 2026-09-29 | (infrastructure) | dry run 2026-09-08..2026-09-29: 6676/309/69/- | n/a | (this commit) | n/a | scripts/monthly_update_openalex.py committed as the standard search path; SOP docs/UPDATE-PROCESS.md; ITX scheduled task SLR-website-monthly-update registered for day 1 of each month 10:07. First scheduled batch: October 2026. |
