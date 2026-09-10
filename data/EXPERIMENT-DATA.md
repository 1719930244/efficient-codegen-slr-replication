# RQ6 Experiment Data Provenance

This document maps every raw-data directory in this package to the machine,
campaign, and software stack that produced it, and records what was lost,
quarantined, or dropped, so that every number in the article is traceable.

## Directory map

| Path | Machine | Campaign | Status |
|---|---|---|---|
| `v100-original-results/composition,pareto,energy_round` | 4x Tesla V100-SXM2-32GB (original campaign server) | Apr 2026, 3 runs per cell | run-level aggregate JSON only, no per-task records |
| `v100-original-results/heplus` | same machine | Sep 7 2026 regeneration pass (P10-P12, C04-C06) | full per-task jsonl |
| `v100-original-results/logs-original-campaign` | same machine | Apr 16-19 2026 | 15 raw campaign logs |
| `v100-original-results/logs-v100-heplus` | same machine | Sep 2026 regeneration | 4 GPU logs + env logs |
| `v100-original-results/provenance` | same machine | Apr 2026 | dependency manifests, run scripts, repo README, all as-found |
| `a800-results/a800-1` | 2x A800-SXM4-40GB (rental server 1) | Aug-Sep 2026 main extension campaign | final state at 2026-09-08 08:47 UTC, machine retired 08:55 UTC |
| `a800-results/a800-2` | 2x A800-SXM4-40GB (rental server 2) | same campaign, second machine | state at 2026-09-08 06:10 UTC, machine retired 14:30 UTC (see losses below) |
| `a800-results/logs-a800-1`, `logs-a800-2` | both | | operational logs; files over 10 MB are trimmed to head 3000 + tail 15000 lines and gzip-compressed, trimming is marked inside each file |
| `xgpu-3090ti-217` | GeForce RTX 3090 Ti 24 GB (lab server, third architecture) | Sep 8 2026, single run per cell | full per-task jsonl + acceptance JSONs + judged artifacts |

## Software stacks (recorded as-run)

| Platform | Python | torch | transformers | bitsandbytes | driver |
|---|---|---|---|---|---|
| V100 original (Apr 2026) | 3.10 (micromamba env `exp`, not preserved) | see note below | see note below | see note below | 570.195.03 (as of Sep 2026) |
| V100 regeneration (Sep 2026) | 3.10.6 | 2.6.0+cu124 | 5.5.4 | 0.49.2 | 570.195.03 |
| A800 (Aug-Sep 2026) | 3.12 | 2.6.0+cu124 | 5.16.1 | 0.50.2 | 590.48.01 |
| RTX 3090 Ti (Sep 2026) | 3.10.20 (pre-existing borrowed env) | 2.5.1+cu124 | 5.9.0 | 0.49.2 | 550.90.07 |

**V100 original-stack note (irreconcilable evidence, archived as-is).** The
article records the original V100 runs as PyTorch 2.5.1 with CUDA 12.4, written
at submission time when the runtime environment was still inspectable. The
runtime environment (micromamba env `exp`) was later deleted. The dependency
manifests archived in `v100-original-results/provenance/` (pyproject.toml and
uv.lock, both dated 2026-04-16 09:35 UTC, roughly 50 minutes before the first
composition run started) pin `torch==2.6.0` with the cu124 index and
`transformers>=5.5.4`. The April run logs confirm the micromamba env was the
actual interpreter, not a uv venv, so the manifests may describe an intended
migration rather than the runtime. Neither source can now be verified against
the deleted environment; both are archived unmodified and the article keeps the
submission-time record.

The Sep 2026 V100 regeneration environment is fully documented by the uv sync
install log `v100-original-results/logs-v100-heplus/v100-regen-uvsync.log`
(complete pinned package list) and `v100-regen-dl05b.log` (draft-model download
checksums).

## Losses at A800 retirement (2026-09-08)

Rental server 1 was backed up in full at 08:47 UTC, eight minutes before its
08:55 UTC expiry; nothing was lost there. Rental server 2 expired at 14:30 UTC;
its last off-machine backup was taken at 06:10 UTC, so the following work
products, all completed on the machine after that backup, were lost:

1. **B02 final judged result** (BigCodeBench instruct protocol, INT8 standard,
   300 tasks). The run had reached 275/300 by 09:16 UTC and completed on the
   machine; only the first 83 task records (state at 06:10 UTC) survive, in
   `a800-results/a800-2/bcb_instruct/B02.jsonl`. The article therefore reports
   the instruct-protocol check over three of the four central cells and says so
   explicitly.
2. **MBPP+ judged values for M01, M02, M04** and the completed M02 generation.
   This loss has no effect on the article: the MBPP+ layer was dropped on a
   pre-declared gate (see below) before these values were inspected.
3. **P15/P13 verification reruns.** The cells they re-verify were already
   judged and reported from earlier complete runs; nothing in the article
   depends on the reruns.
4. **Server 2 orchestrator logs after 06:10 UTC.** Key checkpoints (M03 gate
   value 32.80 base / 28.57 plus; B02 at 275/300; M04 completion) were captured
   in monitoring transcripts at 09:16 UTC and are quoted in this README.

## MBPP+ layer: attempted, gated out, dropped

MBPP+ (EvalPlus judge over MBPP) was attempted on server 2. Three successive
harness defects were found and quarantined under
`a800-results/a800-2/mbppplus/broken_*` (inner truncation of instruct-style
outputs; a completion parser defect; and a wrong generation prompt that omitted
the main-protocol signature scaffolding). After all three fixes, the INT4 cell
(M03) scored base 32.80 / plus 28.57 against a sanity gate of roughly 60 that
was declared before the value was inspected (the main-harness MBPP INT4 score
is 66.20). The gate failed, so the MBPP+ layer was dropped and no MBPP+ number
appears anywhere in the article. The fixed-prompt regenerations that survived
the 06:10 backup are archived as-is for audit.

## Quarantine manifest (defective outputs kept for audit)

| Directory | Defect |
|---|---|
| `a800-1/heplus/broken_detok` | DeepSeek slow-tokenizer decode dropped spaces (fixed by tokenizer self-heal) |
| `a800-1/bcb_instruct/broken_judge`, `broken_judge2`, `a800-2/bcb_instruct/broken_judge2` | judge prepended completion-style prefix to instruct outputs; completions not saved (fixed by dedicated instruct judge) |
| `a800-2/mbppplus/broken_inner_trunc` | generator truncated instruct/chat outputs internally (fixed by `truncate=False`) |
| `a800-2/mbppplus/broken_parser` | completion parser defect |
| `a800-2/mbppplus/broken_prompt` | official EvalPlus prompt without main-protocol signature scaffolding (fixed by rebuilding prompts from `eval_mbpp.build_prompt`) |

## Third-architecture reference point (`xgpu-3090ti-217`)

Generated on the RTX 3090 Ti server on 2026-09-08 (chain log
`xgpu-3090ti.log`, launcher `xgpu_3090ti.sh`, patched script copies under
`scripts/`). The server has no internet egress; the draft model and benchmark
file were relayed in, and generation used a pre-existing environment (stack
table above), which the article discloses as a third distinct software stack.

Config IDs in `results/heplus/`: P10/P11/P12 = FP16/INT8/INT4 standard greedy,
C04/C05/C06 = FP16/INT8/INT4 speculative greedy, all HumanEval, 164 problems,
single run. `results/acceptance/T7*.json` = instrumented draft-verify loop,
60-problem slice, K=5.

Judging ran on a separate lab server (2x V100) with EvalPlus 0.3.1 under
Python 3.10.12, using G-prefixed IDs to avoid collision with the A800 key
space: **G10=P10, G11=P11, G12=P12, G04=C04, G05=C05, G06=C06**. Judged values
(base pass@1 / HumanEval+ pass@1): G10 81.10/76.22, G11 80.49/75.00,
G12 79.88/75.00, G04 81.10/76.22, G05 79.88/75.00, G06 78.66/73.78.
Per-task details and EvalPlus raw results are under `judged/`;
`judged/heplus_summary.json` is the machine-written summary;
`judged/judge_217.log` is the judging transcript. The G04-vs-G10 per-task pass
vectors are identical in all 164 tasks on both base and plus tests, the
losslessness check greedy speculative decoding must satisfy.

Board energy was not measured on this platform: the GeForce driver does not
expose `nvmlDeviceGetTotalEnergyConsumption`.

## Article-to-data quick index

- Table "cross-architecture comparison" (V100/A800/3090 Ti): V100 column from
  `v100-original-results` aggregates + regeneration judged values; A800 column
  from `a800-results/a800-1` (5-run campaign); 3090 Ti column from
  `xgpu-3090ti-217` + `judged/`.
- Acceptance table: A800 rows from `a800-results/a800-1/acceptance/T7*.json`;
  3090 Ti rows from `xgpu-3090ti-217/results/acceptance/T7*.json`.
- Second-family table: `a800-results/a800-1/heplus/F0*.jsonl` and judged
  summaries in `heplus_summary.json` on the same directory.
- BigCodeBench instruct check: `a800-results/a800-1/bcb_instruct/B01,B03,B04`
  (B02 partially lost, see above).
