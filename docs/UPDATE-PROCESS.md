# Companion Website Monthly Update SOP

Cadence: **one batch per calendar month**, first batch October 2026. The
companion website and `data/monthly-updates/` are updated together; the
frozen 122-study corpus and `data/statistics.json` are **never** touched by a
monthly batch. IEEE Xplore, ACM DL, Scopus, and Web of Science remain
deferred to the annual full sweep (institutional credentials).

## Ownership and scheduling

- Executed by the ITX machine's Windows scheduled task
  `SLR-website-monthly-update` (day 1 of each month, 10:07 local, headless
  `claude -p` run; script and logs in `C:\Users\daoge\slr-monthly-update\`).
- Any machine session may run a batch manually by following this SOP, but
  **first check `data/monthly-updates/` for an existing batch of the current
  month** to avoid duplicates. One batch per month, whoever lands it first.
- If a scheduled run fails, the failure is recorded in
  `data/monthly-updates/UPDATE-LOG.md`; the next run (or a manual run) covers
  the accumulated window, so a missed month self-heals.

## Procedure

1. **Window.** Read the latest `data/monthly-updates/*.json`; the new window
   starts the day after its `window` end date and ends today.

2. **Search.** From the repo root (OpenAlex is directly reachable on the
   campus network; no proxy; the script retries 429s):

   ```
   python scripts/monthly_update_openalex.py --start <S> --end <E> \
       --outdir data/monthly-update-<YYYY-MM>
   ```

   Outputs: `candidates-raw.json`, `candidates-boolean.json` (Group B AND
   Group C in title or abstract), `nominated.json` (title-strict efficiency
   signal, deduplicated against `primary-studies.csv` and all prior
   batches), `funnel.json`.

3. **Screening (judgment step).** From `nominated.json`, select the primary
   studies under the original inclusion criteria: an efficiency technique or
   measurement for LLM-based code generation somewhere in the lifecycle, with
   a substantive technical or empirical contribution. Exclude position
   pieces, tutorials, non-LLM systems, studies whose efficiency claim is
   incidental, and anything duplicating the frozen corpus or prior batches
   under a different title. Record the rejected nominees in `dropped`
   (titles only). Assign each selected study an RQ group (RQ1 data, RQ2
   training, RQ3 inference, RQ4 deployment, RQ5 evaluation, or Other).

4. **Batch record.** Write `data/monthly-updates/<YYYY-MM>.json` following
   the 2026-09 schema:

   ```json
   {
     "window": "<S>..<E>",
     "source": "OpenAlex (scripts/monthly_update_openalex.py; three-group schema identical to the original protocol)",
     "funnel": {"candidates": <group_a_hits>, "boolean": <boolean_pass>, "nominated": <n>, "selected": <k>},
     "studies": [{"openalex_id": "...", "title": "...", "rq": "RQ3"}],
     "dropped": ["title", "..."],
     "note": "post-submission coverage update; frozen 122-study corpus unchanged"
   }
   ```

5. **Website render** (`docs/index.html`):
   - Insert a new `update-batch` card **above** the newest existing card
     (batches are listed newest first). Copy the September 2026 card markup.
     `h3` = "<Month> <Year> &mdash; <k> new studies"; `batch-meta` = window,
     source, funnel with arrows, credential-deferral sentence, and an overlap
     note if any study also appears in an earlier batch (mark it inline too).
     Group studies by RQ with `update-rq` blocks; each item links its
     OpenAlex URL. End with the structured-record link to the batch JSON.
   - Banner: "Last content update: <strong><Month> <Year></strong>".
   - Footer: "Last updated: <Month> <Year>".
   - Living Updates intro: refresh the cumulative unique-study count
     (subtract cross-batch overlaps).
   - Validate tag balance:
     `python -c "from html.parser import HTMLParser; ..."` or any parser;
     zero mismatched tags required.

6. **Log.** Append one row to `data/monthly-updates/UPDATE-LOG.md`: date,
   window, funnel, commit hash, Pages verification result, anomalies.

7. **Commit and push.** Descriptive commit message, no Co-Authored-By.
   GitHub is not directly reachable from the campus network; use the sg
   tunnel:

   ```
   ssh -f -N -D 10808 sg
   git -c http.https://github.com.proxy=socks5h://127.0.0.1:10808 push origin master
   ```

   Kill the tunnel afterwards (locate the `ssh.exe ... -D 10808 sg` process
   with `powershell Get-CimInstance Win32_Process` and `taskkill /PID <pid> /F`).
   Never force-push. If the push fails, keep the local commit and log
   "push pending"; the next run pushes accumulated commits.

8. **Verify Pages** (build takes about a minute):

   ```
   ssh sg "curl -s https://1719930244.github.io/efficient-codegen-slr-replication/ | grep -c '<Month> <Year>'"
   ```

## Boundaries

- Monthly batches never modify: `data/primary-studies.csv`,
  `data/statistics.json`, `data/by-rq/`, the manuscript, or the frozen-corpus
  claims anywhere on the site.
- Website numbers must stay consistent with the manuscript's audited canon
  (head counts 25/24/59/24/21, 122 studies, 6 RQs, 8 findings). If the
  manuscript changes, the site overview changes in the same batch.
- Study titles/links in cards come from the batch JSON verbatim; no
  paraphrasing.
