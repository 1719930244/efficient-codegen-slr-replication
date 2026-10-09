# Supplementary coverage refresh -- arXiv screening log

Window 2026-04-01 to 2026-08-31. Source: arXiv API refresh (2026-09-29), identical three-group schema to the OpenAlex pass (2026-09-07). Candidates: 111 new title-strict nominations (funnel.json), deduplicated by normalized title against the frozen 122-study corpus, the 2026-09 OpenAlex batch, and its dropped list.

Inclusion/exclusion follows Table tab:inclusion_exclusion (IC1-IC4, EC1-EC5): an efficiency technique or efficiency measurement for LLM-based code generation somewhere in the model lifecycle, with a substantive technical or empirical contribution. Excluded: position pieces, surveys/taxonomies, tutorials, non-LLM work, incidental efficiency claims, non-code-generation tasks, output-artifact-runtime-efficiency techniques (kept only as sparse boundary benchmarks, per the DSEffi-Bench precedent), and duplicates of corpus/batch studies under another title.

Buckets: RQ1 data / RQ2 training / RQ3 inference / RQ4 deployment / RQ5 evaluation / cross-stage.

| # | arXiv | date | decision | bucket | reason |
|---|-------|------|----------|--------|--------|
| 000 | 2609.00237v2 | 2026-08-31 | EXCLUDE | - | general multi-agent routing; efficiency on mixed reasoning+code benchmarks, not a code-generation-specific technique |
| 001 | 2609.00058v1 | 2026-08-30 | EXCLUDE | - | targets runtime efficiency of generated CUDA kernels (output artifact), application-specific, not generation-process efficiency |
| 002 | 2608.27513v1 | 2026-08-27 | EXCLUDE | - | general model mixed-precision quantization, not code-specific |
| 003 | 2608.26374v1 | 2026-08-26 | EXCLUDE | - | general diffusion LM length control, not code generation |
| 004 | 2608.26049v1 | 2026-08-26 | EXCLUDE | - | security defense for poisoned RTL models; efficiency not the objective (EC1) |
| 005 | 2608.23632v1 | 2026-08-23 | EXCLUDE | - | code preference optimization for correctness; efficiency not a design objective (EC1) |
| 006 | 2608.09745v1 | 2026-08-10 | EXCLUDE | - | general self-distillation method, not code-specific |
| 007 | 2609.11956v1 | 2026-08-06 | EXCLUDE | - | offline post-training of code LLMs but substantially duplicates retained wu2026offlinerl |
| 008 | 2608.04450v1 | 2026-08-05 | EXCLUDE | - | benchmark of generated GPU-communication code efficiency (boundary); redundant with retained CodegenBench under the RQ5 cap |
| 009 | 2608.04336v1 | 2026-08-05 | EXCLUDE | - | quality-cost config search for code gen; overlaps existing routing and adaptive-sampling coverage, marginal addition |
| 010 | 2608.02942v1 | 2026-08-03 | EXCLUDE | - | general few-step diffusion LM distillation, not code |
| 011 | 2608.01927v1 | 2026-08-03 | INCLUDE | RQ3 inference | efficient on-demand partial-dependency-graph context retrieval for repo-level code gen; 7.4x faster, higher Pass@1 |
| 012 | 2608.01667v1 | 2026-08-03 | EXCLUDE | - | general turn-level RL credit assignment, not code-specific |
| 013 | 2608.00909v1 | 2026-08-02 | EXCLUDE | - | latency-aware financial hardware generation; output-artifact efficiency, niche |
| 014 | 2607.29626v1 | 2026-07-31 | EXCLUDE | - | agents as hyperparameter optimizers; not code generation (EC2) |
| 015 | 2608.02641v1 | 2026-07-31 | EXCLUDE | - | optimization autoformulation, not code-generation efficiency |
| 016 | 2608.13596v1 | 2026-07-31 | EXCLUDE | - | general cross-scale pruning/knowledge transfer, not code |
| 017 | 2607.26805v1 | 2026-07-29 | EXCLUDE | - | efficient repo-level context selection but overlaps retained liu2026dyretriever (same problem) |
| 018 | 2607.24051v1 | 2026-07-27 | EXCLUDE | - | astrodynamics trajectory optimization agent, not code generation |
| 019 | 2607.21677v1 | 2026-07-23 | EXCLUDE | - | narrow radio-astronomy code-optimization case study; output-artifact efficiency |
| 020 | 2607.20806v1 | 2026-07-23 | EXCLUDE | - | general lightweight-LLM profiling, not code-generation-specific |
| 021 | 2607.20630v1 | 2026-07-22 | EXCLUDE | - | DB query-processing code generation demo; output efficiency, niche |
| 022 | 2607.19450v3 | 2026-07-21 | EXCLUDE | - | general expert-to-generalist distillation, not code |
| 023 | 2607.16850v1 | 2026-07-18 | EXCLUDE | - | general entropy-controlled policy optimization, not code |
| 024 | 2607.11505v2 | 2026-07-13 | EXCLUDE | - | general on-policy distillation, not code |
| 025 | 2607.11012v1 | 2026-07-13 | EXCLUDE | - | general on-policy distillation framework, not code |
| 026 | 2607.10210v1 | 2026-07-11 | EXCLUDE | - | code watermarking quality/detectability; efficiency not the objective (EC1) |
| 027 | 2607.08010v2 | 2026-07-09 | EXCLUDE | - | general self-evolving agent tool-making; code incidental |
| 028 | 2607.07643v1 | 2026-07-08 | EXCLUDE | - | automated HLS for FPGAs; hardware output-artifact efficiency |
| 029 | 2607.07554v1 | 2026-07-08 | EXCLUDE | - | quantum circuit synthesis (quant-ph), not code generation |
| 030 | 2607.06519v1 | 2026-07-07 | EXCLUDE | - | general long-context KV cache compression, not code |
| 031 | 2607.05121v1 | 2026-07-06 | EXCLUDE | - | prompt-rule evolution for code correctness/repair; efficiency not the objective (EC1) |
| 032 | 2607.04428v1 | 2026-07-05 | EXCLUDE | - | general on-policy self-distillation for diffusion LMs, not code |
| 033 | 2607.03328v2 | 2026-07-03 | EXCLUDE | - | CV/IR native hash learning, not code |
| 034 | 2607.00939v1 | 2026-07-01 | EXCLUDE | - | quantum application generation for test optimization, not code-gen efficiency |
| 035 | 2607.00254v1 | 2026-06-30 | EXCLUDE | - | query-centric AI-workflow optimization (DB), not code generation |
| 036 | 2606.31732v1 | 2026-06-30 | EXCLUDE | - | visual-to-code correctness/quality; efficiency incidental |
| 037 | 2606.29239v3 | 2026-06-28 | EXCLUDE | - | quantization-backdoor repair (security), not efficiency |
| 038 | 2606.28962v1 | 2026-06-27 | EXCLUDE | - | quantization-backdoor defense (security), not efficiency |
| 039 | 2606.27733v3 | 2026-06-26 | EXCLUDE | - | robust bash code gen; robustness/correctness objective, not efficiency (EC1) |
| 040 | 2606.26453v2 | 2026-06-24 | EXCLUDE | - | CUDA kernel optimization; output-artifact runtime efficiency, application-specific |
| 041 | 2606.25519v2 | 2026-06-24 | EXCLUDE | - | quantization token-inflation in general reasoning models, not code |
| 042 | 2606.23104v1 | 2026-06-22 | EXCLUDE | - | general on-policy distillation reweighting, not code |
| 043 | 2606.18023v1 | 2026-06-16 | INCLUDE | RQ3 inference | parallel-loop 7B coder architecture; loop count traded against latency and KV-cache memory for test-time scaling |
| 044 | 2606.16871v1 | 2026-06-15 | EXCLUDE | - | general AI-agent evaluation, not code-generation efficiency |
| 045 | 2606.16534v2 | 2026-06-15 | EXCLUDE | - | narrow Julia-on-supercomputers empirical case; output-artifact scalability |
| 046 | 2606.15453v1 | 2026-06-13 | EXCLUDE | - | general MoE expert prefetching inference system, not code |
| 047 | 2606.12370v1 | 2026-06-10 | EXCLUDE | - | general RL-training acceleration via MTP, not code |
| 048 | 2606.10334v1 | 2026-06-09 | EXCLUDE | - | code+visual self-distillation for quality; efficiency incidental |
| 049 | 2606.09956v1 | 2026-06-08 | EXCLUDE | - | bug classification, not code generation (EC2) |
| 050 | 2606.08944v1 | 2026-06-08 | EXCLUDE | - | long-context RTL optimization; output-artifact efficiency, application |
| 051 | 2606.06826v1 | 2026-06-05 | EXCLUDE | - | trains for runtime-efficient OUTPUT code; output-artifact efficiency outside generation-process scope |
| 052 | 2606.06821v1 | 2026-06-05 | EXCLUDE | - | structured skeleton supervision for efficient OUTPUT code; output-artifact efficiency, near-duplicate of #051 |
| 053 | 2606.09885v1 | 2026-06-03 | EXCLUDE | - | general MoE expert-neuron pruning, not code |
| 054 | 2606.02963v1 | 2026-06-01 | EXCLUDE | - | cross-platform kernel generation; output-artifact kernel efficiency, application |
| 055 | 2606.04023v1 | 2026-06-01 | INCLUDE | RQ5 evaluation | benchmark measuring efficiency of generated parallel code across x86/Sunway/Kunpeng; boundary benchmark (extends DSEffi to hardware diversity), open-source |
| 056 | 2606.01249v3 | 2026-05-31 | EXCLUDE | - | general trust-region on-policy distillation, not code |
| 057 | 2605.29734v1 | 2026-05-28 | EXCLUDE | - | operator-optimization memory mechanism, not code generation |
| 058 | 2605.29716v1 | 2026-05-28 | EXCLUDE | - | noise-aware LoRA PEFT for general diffusion LLMs, not code |
| 059 | 2605.26842v2 | 2026-05-26 | EXCLUDE | - | general Muon optimizer for LM training, not code |
| 060 | 2605.26646v1 | 2026-05-26 | EXCLUDE | - | general multi-agent RL optimization framework, not code |
| 061 | 2605.25246v3 | 2026-05-24 | EXCLUDE | - | OR algorithm-design benchmark, not code-generation efficiency |
| 062 | 2605.23273v1 | 2026-05-22 | EXCLUDE | - | engineering topology optimization via agents, not code generation |
| 063 | 2605.22817v1 | 2026-05-21 | EXCLUDE | - | general diversity-training RL for test-time search, not code |
| 064 | 2605.22675v1 | 2026-05-21 | EXCLUDE | - | general self-policy distillation, not code |
| 065 | 2605.20643v1 | 2026-05-20 | EXCLUDE | - | general adaptive-view self-distillation, not code |
| 066 | 2605.19102v1 | 2026-05-18 | EXCLUDE | - | RL prompt optimization for code correctness (Pass@1); efficiency not the objective (EC1) |
| 067 | 2605.14718v1 | 2026-05-14 | EXCLUDE | - | FHE-on-TPU code optimization; output-artifact efficiency, crypto niche |
| 068 | 2605.14539v1 | 2026-05-14 | EXCLUDE | - | general correction-oriented policy optimization, not code-efficiency |
| 069 | 2605.23966v1 | 2026-05-12 | EXCLUDE | - | optimization-modeling validation framework, not code generation |
| 070 | 2607.16206v1 | 2026-05-08 | EXCLUDE | - | general exploratory RL framework, not code |
| 071 | 2605.30359v2 | 2026-05-08 | EXCLUDE | - | evolutionary kernel optimizer; output-artifact kernel efficiency |
| 072 | 2605.06443v1 | 2026-05-07 | EXCLUDE | - | wireless precoding optimization, not code generation |
| 073 | 2605.08134v1 | 2026-05-01 | EXCLUDE | - | general diffusion LM activation-reuse inference, not code |
| 074 | 2604.27296v1 | 2026-04-30 | INCLUDE | RQ3 inference | token-efficient adaptive edit-format (diff vs full code); matches accuracy, cuts latency/cost >30% on code editing |
| 075 | 2604.27115v1 | 2026-04-29 | EXCLUDE | - | general task-specific pruning/collapse study, not code |
| 076 | 2604.26951v1 | 2026-04-29 | EXCLUDE | - | general cross-architecture distillation for diffusion LLMs, not code |
| 077 | 2604.25903v1 | 2026-04-28 | EXCLUDE | - | green compression pipeline for general LMs, not code-specific |
| 078 | 2604.24927v2 | 2026-04-27 | EXCLUDE | - | general latent-distilling exploration, not code |
| 079 | 2604.24647v1 | 2026-04-27 | EXCLUDE | - | general layer-dependent KV pruning long-context, not code |
| 080 | 2604.23892v1 | 2026-04-26 | EXCLUDE | - | analytics-informed performance-optimization framework, not code-generation efficiency |
| 081 | 2604.23623v1 | 2026-04-26 | EXCLUDE | - | large+small LM efficient reasoning, not code |
| 082 | 2604.23002v1 | 2026-04-24 | EXCLUDE | - | Lean autoformalisation at scale; efficiency incidental |
| 083 | 2604.21794v1 | 2026-04-23 | EXCLUDE | - | general end-to-end multi-agent system optimization, not code |
| 084 | 2604.21952v1 | 2026-04-23 | EXCLUDE | - | focus-session overview of acceleration techniques; overview/position (EC4) |
| 085 | 2604.17708v2 | 2026-04-20 | EXCLUDE | - | general co-evolving optimization agents, not code |
| 086 | 2604.17351v1 | 2026-04-19 | EXCLUDE | - | simulator construction via bilevel optimization; code incidental |
| 087 | 2604.17227v1 | 2026-04-19 | EXCLUDE | - | cloud-native LLM systems research agenda; position/agenda (EC4) |
| 088 | 2605.16299v2 | 2026-04-17 | EXCLUDE | - | self-evolving coding via adversarial tests; correctness/scalability objective, not a measured efficiency metric (EC1) |
| 089 | 2604.15642v1 | 2026-04-17 | EXCLUDE | - | simulated-annealing hardware-design code gen; output PPA efficiency, application |
| 090 | 2604.15001v2 | 2026-04-16 | EXCLUDE | - | RTL correctness+PPA co-optimization; output-artifact efficiency, application |
| 091 | 2604.13010v3 | 2026-04-14 | EXCLUDE | - | offline on-policy distillation for general reasoning models, not code |
| 092 | 2605.04078v2 | 2026-04-14 | EXCLUDE | - | general validity-calibrated reasoning distillation, not code |
| 093 | 2604.12290v2 | 2026-04-14 | EXCLUDE | - | self-evolving-agent engineering-task benchmark, not code-gen efficiency |
| 094 | 2604.10387v2 | 2026-04-12 | EXCLUDE | - | LLM-derived GPU thread mapping; output-artifact efficiency, niche |
| 095 | 2604.04809v2 | 2026-04-06 | EXCLUDE | - | taxonomy of software energy smells; survey/taxonomy (EC4) |
| 096 | 2604.03632v1 | 2026-04-04 | EXCLUDE | - | cross-attempt state reuse for repo gen quality; efficiency not a measured objective |
| 097 | 2604.02985v1 | 2026-04-03 | EXCLUDE | - | prompt compression for general LLM/RAG inference, not code generation (EC2) |
| 098 | 2604.02492v1 | 2026-04-02 | EXCLUDE | - | token-efficient multimodal reasoning, not code |
| 099 | 2604.02007v3 | 2026-04-02 | EXCLUDE | - | RL post-training for general efficient reasoning, not code |
| 100 | 2607.29637v1 | 2026-07-31 | EXCLUDE | - | adaptive visual compression for code UNDERSTANDING, not generation (EC2) |
| 101 | 2605.28510v2 | 2026-05-27 | EXCLUDE | - | provenance/plagiarism tracking of generated code; efficiency of the detector, not generation (EC2) |
| 102 | 2605.23200v1 | 2026-05-22 | EXCLUDE | - | general long-context KV compression for reasoning, not code |
| 103 | 2608.15117v1 | 2026-08-15 | INCLUDE | RQ4 deployment | measurement study of peak-VRAM stability/forecasting for 4-bit quantized code-synthesis agents; released corpus |
| 104 | 2605.21082v1 | 2026-05-20 | EXCLUDE | - | GUI-automation code synthesis; efficiency of the automation task, not the generation process |
| 105 | 2607.14709v1 | 2026-07-16 | EXCLUDE | - | programmatic distillation for financial reasoning over tables, not code-gen efficiency |
| 106 | 2606.14581v5 | 2026-06-12 | EXCLUDE | - | chemistry reaction-optimization ranking, not code |
| 107 | 2605.09985v1 | 2026-05-11 | EXCLUDE | - | cognitive-science abstraction learning, not code |
| 108 | 2605.08037v1 | 2026-05-08 | EXCLUDE | - | general preference-graph optimization, not code |
| 109 | 2605.05485v1 | 2026-05-06 | INCLUDE | RQ3 inference | compiles reasoning traces into symbolic solvers for zero-token test-time program synthesis; 78% token reduction, code/data released |
| 110 | 2604.05072v2 | 2026-04-06 | EXCLUDE | - | SVG/vector-graphics program modeling, not code generation |

## Summary
- Screened: 111
- Included before cap: 6 (indices [11, 43, 55, 74, 103, 109])
- All 6 fall within the per-bucket cap of five over the combined two-pass set (see batch-2026-08-31.json), so none are dropped at the cap stage.
- Retained additions: 011 liu2026dyretriever (RQ3), 043 yang2026loopcoder (RQ3), 074 cheng2026adaedit (RQ3), 109 naik2026reacomp (RQ3), 103 banerjee2026quantagent (RQ4), 055 li2026codegenbench (RQ5).
