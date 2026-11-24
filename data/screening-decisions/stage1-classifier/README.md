# Stage-1 classifier

`classify.py` is the rule-based title-and-abstract classifier of Stage 1
(Section 3.3 of the manuscript). It matches the code-generation, efficiency,
and LLM vocabularies, together with lists of other-task and survey terms,
against the title and abstract of each record, counts the matches per group,
and assigns one of three labels:

- **Likely-Include**: code-generation and LLM terms, and no term from other tasks.
- **Uncertain**: code generation and efficiency without LLM terms; efficiency
  and LLM without code-generation terms; or code generation and LLM together
  with terms from other tasks.
- **Likely-Exclude**: every other record, including records published before
  2017, surveys without code-generation terms, and records with two or more
  other-task terms and no code-generation terms.

The rule uses no tuned threshold or weights.

## Files

| File | Contents |
|---|---|
| `classify.py` | The classifier. Place the deduplicated search export as `deduplicated.bib` in the directory named by `SLR_DATA_DIR` (default `./data`) and run `python classify.py`. |
| `stage1-labels.csv` | The Stage-1 label and matched-term counts of each record in the export of the main search (2026-03-06): 22,119 rows, in which five records appear twice. |
| `verify_labels.py` | Re-derives every label in `stage1-labels.csv` from its matched-term counts with the same rule and reports any difference. |

```bash
python verify_labels.py
# records: 22119
#   Likely-Exclude   18960
#   Likely-Include    1380
#   Uncertain         1779
# labels that differ from the rule: 0
```

Records from the update searches (April to October 2026) were screened
through the funnels documented in `docs/search-queries.md`. The final 3,840
Stage-1 candidates also include records that the authors retained during
screening.
