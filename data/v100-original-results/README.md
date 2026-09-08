# V100 original campaign results (recovered)

Provenance: the original experiments ran on a 4x Tesla V100-SXM2-32GB server that left
the group in July 2026. The per-run aggregate result files were recovered from the
retired machine on 2026-09-07 and are archived here unchanged.

Contents:
- `composition/run{1,2,3}/C01-C12.json` - technique composition aggregates per run
- `pareto/run{1,2,3}/P01-P15.json` - Pareto sweep aggregates per run (5 scales x 3 precisions)
- `energy_round/{composition,pareto}/` - dedicated single-pass NVML energy measurements
- `heplus/` - HumanEval+ regeneration on the same V100 server (2026-09-07, identical
  harness to the A800 side) with generation logs under `logs-v100-heplus/`

Notes:
- Aggregate files carry run-level means, percentiles, memory, and energy; the `details`
  arrays are empty, i.e. no per-task records were retained on the V100 side.
- Spot checks reproduce the published V100 numbers exactly (e.g. C01 pass@1 81.71,
  INT8 latency ratio 4.38, 1.5B FP16 46.34, 14B INT4 88.41).

Update 2026-09-08: the six V100 HumanEval+ regenerations (`heplus/P10,P11,P12,C04,C05,C06`)
were judged offline with EvalPlus 0.3.1; judged values (base and plus pass@1) are recorded in
`heplus/v100_heplus_judged_summary.json` under V-prefixed keys (V10=P10, V11=P11, V12=P12,
V04=C04, V05=C05, V06=C06) to distinguish them from the A800 regeneration of the same IDs.
