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
