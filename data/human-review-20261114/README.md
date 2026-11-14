# Human review of the screening samples (2026-11-14)

Two authors reviewed the same three samples in a Yes/No form: each record
carried a proposed decision and a written reason, and each author confirmed
it or objected. Nothing else was required of them.

## Files

| File | Rows | Content |
|------|------|---------|
| `human-review-770.csv` | 770 | One row per record: source, id, proposed decision, reason, both authors' confirmation, override, final decision |
| `agreement.json` | — | Confirmation counts, agreement, kappa, and the final coverage distribution |

Samples: `Coverage-400` (400 records), `TA-200` (200 records), `FT-170` (170
records). The records and the proposed decisions are the same files used by
the automated screening passes, so every row can be traced to
`data/coverage-audit-20261007/`, `data/screening-decisions/dual-screening-kit/`,
and `data/screening-decisions/original-2026-03/`.

## Result

| | Rater 1 | Rater 2 |
|---|---|---|
| Confirm (Yes) | 739 | 724 |
| Object (No) | 31 | 46 |
| Override recorded | 0 | 0 |

Agreement 98.1%, kappa 0.795 over the Yes/No decision. Fifteen records drew
opposite answers from the two authors. No author recorded an alternative
decision, so the final decision for every record is the proposed one.

Objections by sample:

| Sample | Rater 1 | Rater 2 |
|--------|---------|---------|
| Coverage-400 | 13 | 21 |
| TA-200 | 13 | 17 |
| FT-170 | 5 | 8 |

Coverage-400 after review: 3 Include, 32 Uncertain, 365 Exclude. The
projection of eligible studies outside the corpus is therefore unchanged at
about 28, with a recall of about 0.78.

## What this review supports

It supports the statement that the authors read every sampled record and
confirmed or disputed the proposed decision. It does not support a claim of
independent dual annotation for the full corpus, because the two authors
reviewed the same proposal rather than two independent decisions from
scratch. The kappa above describes the Yes/No confirmation step only.