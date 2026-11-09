#!/usr/bin/env python3
"""
Apply expert fine-grained classification to all 123 primary studies.
Each paper is manually classified based on title + abstract + screening reason.
"""

import csv
import json
from pathlib import Path

BASE = Path("/home/szw/overleaf/efficient-codegen-survey/data")
OUTPUT_CSV = BASE / "fine-grained-classification.csv"

# ── Expert Classification ──────────────────────────────────────────────────
# Format: key -> (primary_categories, secondary_categories, scope_flags, rq_assignment, rationale)

CLASSIFICATIONS = {
    # === 2026 papers ===
    "2601_09703": (["3l-CodeSpecific", "3d-PromptCompression"], [], [], "RQ3",
        "Token-efficient syntax optimization (ShortCoder) reduces tokens during code generation"),
    "2601_20223": (["3i-Routing"], [], [], "RQ3",
        "Control models trigger/filter LLM inference to reduce unnecessary code completion requests"),
    "2601_21469": (["3k-MultiAgent"], [], [], "RQ3",
        "Multi-agent collaboration with confidence gating to reduce SLM inference cost"),
    "2601_22132": (["3i-Routing"], [], [], "RQ3",
        "LLM shepherding: strong model provides hints to weak model, reducing full inference cost"),
    "2601_23254": (["3e-ContextPruning"], [], [], "RQ3",
        "Lightweight index-free lexical retrieval optimization for repo-level code completion"),
    "2602_02597": (["3k-MultiAgent"], [], ["SCOPE-Borderline"], "RQ3",
        "Multi-agent context compression for systems code optimization; generation search efficiency but target is code performance"),
    "2602_05712": (["4b-EmpiricalStudy", "3b-EarlyExit"], [], [], "RQ3; RQ4",
        "Empirical energy analysis of LLM inference for code + babbling suppression technique"),
    "2602_07670": (["3j-Sampling"], [], ["SCOPE-Borderline"], "RQ3",
        "Compute-optimal test-time strategies for kernel generation; reports sampling cost but target is kernel performance"),
    "2602_12049": (["2f-RLTraining"], [], ["SCOPE-GenEffCode"], "RQ2",
        "RL with runtime GFLOPS rewards - primary focus is generated HPC code performance, not generation efficiency"),
    "2602_12305": (["3k-MultiAgent"], [], ["SCOPE-GenEffCode"], "RQ3",
        "CUDA kernel optimization framework - primary focus is runtime performance of generated kernels"),
    "2602_15843": (["3d-PromptCompression"], [], [], "RQ3",
        "Task-aware prompt compression validated on code generation benchmarks"),
    "2602_17100": (["3k-MultiAgent"], [], [], "RQ3",
        "RL-optimized multi-agent topology reducing token cost for competition-level code generation"),
    "2602_17169": (["3n-PromptEng"], [], ["SCOPE-GenEffCode"], "RQ3",
        "SimulatorCoder uses ICL/CoT to generate simulator code; focus is generated code performance"),
    "2602_17684": (["2f-RLTraining", "3j-Sampling"], [], [], "RQ2; RQ3",
        "Execution-free reward model for RL training + test-time inference scaling for code generation"),
    "2602_21800": (["4b-EmpiricalStudy"], [], [], "RQ4",
        "Evaluation study of positional embeddings and efficient attention for long-code completion"),
    "2602_24286": (["2f-RLTraining"], [], ["SCOPE-GenEffCode"], "RQ2",
        "Agentic RL for CUDA kernel generation; focus is generating high-performance kernels"),
    "2603_04646": (["3i-Routing"], [], ["SCOPE-Borderline"], "RQ3",
        "Two-stage model escalation (small->large LLM) for Verilog generation with latency tradeoff"),

    # === 2025 papers ===
    "2501_01054": (["2f-RLTraining"], ["3j-Sampling"], [], "RQ2",
        "Dynamic scaling of unit tests for code reward modeling to improve RL training efficiency"),
    "2501_05248": (["3h-ModelPruning"], [], [], "RQ3",
        "Unstructured pruning (Wanda) to derive coding-specific compact LLM sub-models"),
    "2501_05899": (["3n-PromptEng", "4b-EmpiricalStudy"], [], [], "RQ3; RQ4",
        "Evaluates prompt engineering techniques for reducing energy consumption of LLM code generation"),
    "2501_11006": (["3b-EarlyExit"], [], [], "RQ3",
        "GREEN-CODE: RL-based early exit framework for energy-efficient LLM code generation"),
    "2501_18653": (["3k-MultiAgent"], [], [], "RQ3",
        "Cogito multi-agent framework reducing computational cost for code generation"),
    "2501_18657": (["3d-PromptCompression", "2b-PEFT"], [], [], "RQ2; RQ3",
        "Symbolic token compression + PEFT for LLM code generation efficiency"),
    "2502_17139": (["3e-ContextPruning", "3l-CodeSpecific"], [], [], "RQ3",
        "FastCoder: code-aware retrieval acceleration and verification for repo-level code generation"),
    "2502_18489": (["3n-PromptEng"], [], ["SCOPE-GenEffCode"], "RQ3",
        "LLM4EFFI focuses on generating more runtime-efficient code, not on generation process efficiency"),
    "2503_07103": (["3g-PTQ", "4b-EmpiricalStudy"], [], [], "RQ3; RQ4",
        "Evaluates post-training quantization of LLMs for code generation with memory footprint analysis"),
    "2503_17793": (["1a-DataSelection", "1b-DataQuality", "2a-EfficientPretrain"], [], [], "RQ1; RQ2",
        "MoE architecture + program-analytics-based data curation for efficient code LLM"),
    "2503_19876": (["3m-SystemOpt"], [], [], "RQ3",
        "SLA-aware serving framework for code completion with TTFT/E2E latency optimization"),
    "2504_03723": (["3k-MultiAgent"], [], ["SCOPE-Borderline"], "RQ3",
        "VFlow optimizes agentic workflows for Verilog generation with token cost/ROI tradeoff"),
    "2504_12687": (["1a-DataSelection", "2a-EfficientPretrain"], [], [], "RQ1; RQ2",
        "Data selection strategy + dynamic pack tokenization for data-efficient LLM fine-tuning"),
    "2504_14560": (["3o-CoTOpt", "2b-PEFT"], [], ["SCOPE-Borderline"], "RQ2; RQ3",
        "Adaptive reasoning reducing token consumption for Verilog generation + two-stage PEFT training"),
    "2505_01572": (["3a-SpeculativeDecoding"], [], [], "RQ3",
        "PipeSpec: hierarchical pipeline for speculative decoding with code generation evaluation"),
    "2505_14174": (["3n-PromptEng", "3j-Sampling"], [], [], "RQ3",
        "N-rep consistency: cost-efficient text-to-SQL without CoT, using multiple schema representations"),
    "2505_20182": (["3i-Routing"], [], [], "RQ3",
        "Empirical study of strong-weak model collaboration/routing for cost-efficient repo-level code gen"),
    "2506_01604": (["3n-PromptEng", "4b-EmpiricalStudy"], [], [], "RQ3; RQ4",
        "Analyzes prompt patterns to reduce interaction iterations in AI-assisted code generation"),
    "2506_03006": (["2f-RLTraining"], [], ["SCOPE-GenEffCode"], "RQ2",
        "DPO preference alignment for gas-efficient Solidity code; focus is generated code gas efficiency"),
    "2506_09092": (["2f-RLTraining"], [], ["SCOPE-GenEffCode"], "RQ2",
        "CUDA-LLM: FSR framework for generating high-performance CUDA kernels; focus is kernel runtime"),
    "2506_10525": (["3i-Routing"], [], [], "RQ3",
        "AdaptiveLLM: dynamic LLM selection based on CoT length for cost-efficient code generation"),
    "2506_12355": (["3l-CodeSpecific"], [], ["SCOPE-GenEffCode"], "RQ3",
        "QiMeng-Attention: LLM generates efficient attention operators via domain-specific thinking language"),
    "2506_18796": (["3m-SystemOpt"], [], [], "RQ3",
        "Context-aware CodeLLM eviction strategy for efficient model serving under resource constraints"),
    "2507_23194": (["4a-Benchmark"], [], ["SCOPE-GenEffCode"], "RQ4",
        "Geak: Triton kernel AI agent + evaluation benchmarks; focus is kernel speedup"),
    "2508_03258": (["3m-SystemOpt", "3i-Routing"], [], [], "RQ3",
        "SmartLLMs Scheduler: dynamic scheduling and LLM selection for cost-effective code generation"),
    "2508_05970": (["3e-ContextPruning"], [], [], "RQ3",
        "Impact-driven context filtering for cross-file code completion reducing prompt length"),
    "2508_13666": (["3l-CodeSpecific", "4b-EmpiricalStudy"], [], [], "RQ3; RQ4",
        "Empirical study showing code formatting tokens waste LLM budget; proposes removal strategies"),
    "2508_20124": (["2f-RLTraining"], [], ["SCOPE-GenEffCode"], "RQ2",
        "RL framework to improve RUNTIME efficiency of generated code, not generation process efficiency"),
    "2509_09947": (["3n-PromptEng", "4b-EmpiricalStudy"], [], ["SCOPE-Borderline"], "RQ3; RQ4",
        "Evaluates prompt strategies for SLM code generation; reports both generation energy and code runtime"),
    "2509_12159": (["3d-PromptCompression", "3l-CodeSpecific"], [], [], "RQ3",
        "EfficientUICoder: input image token compression + output code token compression for UI code gen"),
    "2509_14093": (["3o-CoTOpt"], [], [], "RQ3",
        "Adaptive CoT compression reducing reasoning length/latency for code generation"),
    "2509_17489": (["2c-Distillation"], [], [], "RQ2",
        "MapCoder-Lite: distilling multi-agent coding knowledge into single 7B model via agent-wise LoRA"),
    "2509_25243": (["3o-CoTOpt", "2f-RLTraining"], [], [], "RQ2; RQ3",
        "RL-guided Chain-of-Draft for token-efficient code generation reasoning"),
    "2510_00446": (["3d-PromptCompression"], [], [], "RQ3",
        "LongCodeZip: code-specific long context compression with dual-stage strategy"),
    "2510_16996": (["3k-MultiAgent"], [], ["SCOPE-GenEffCode"], "RQ3",
        "STARK: multi-agent GPU kernel optimization; primary focus is kernel runtime speedup"),
    "2510_18165": (["3c-NAR", "3j-Sampling"], [], [], "RQ3",
        "Saber: efficient sampling for diffusion language models in code generation"),
    "2510_19644": (["3e-ContextPruning"], [], [], "RQ3",
        "LlavaCode: compressed code representations for RAG reducing TTFT latency"),
    "2511_01884": (["3k-MultiAgent"], [], ["SCOPE-GenEffCode"], "RQ3",
        "CudaForge: multi-agent CUDA kernel optimization; primary focus is kernel runtime performance"),
    "2511_23321": (["2b-PEFT", "3i-Routing"], [], [], "RQ2; RQ3",
        "Chart2Code-MoLA: MoE+LoRA with complexity-aware routing for chart-to-code generation"),
    "2512_08266": (["3l-CodeSpecific"], [], [], "RQ3",
        "Token Sugar: semantic-level shorthand reducing token count for code LLMs"),
    "2512_14048": (["3o-CoTOpt", "3i-Routing"], [], [], "RQ3",
        "Intention CoT with difficulty-aware dynamic routing for efficient code generation"),
    "2512_23367": (["3g-PTQ"], [], [], "RQ3",
        "Post-training quantization of OpenPangu models for deployment on Ascend NPUs"),
    "Giagnorio2025quantizing": (["3g-PTQ", "4b-EmpiricalStudy"], [], [], "RQ3; RQ4",
        "Differentiated replication study evaluating quantization of LLMs for code generation"),
    "He2025codepromptzip": (["3d-PromptCompression"], [], [], "RQ3",
        "CODEPROMPTZIP: code-specific prompt compression for retrieval-augmented code generation"),
    "Zhao2025codeswift": (["3l-CodeSpecific"], [], [], "RQ3",
        "CodeSwift: LLM inference acceleration tailored for code generation"),
    "arxiv_2505_02133": (["3k-MultiAgent", "4b-EmpiricalStudy"], [], [], "RQ3; RQ4",
        "Systematic evaluation of multi-agent collaboration and debugging for code generation latency"),
    "arxiv_2505_14200": (["3g-PTQ", "4b-EmpiricalStudy"], [], ["SCOPE-Borderline"], "RQ3; RQ4",
        "Studies quantization effects on trojans in code LLMs; security focus but reports quantization impacts"),
    "arxiv_2506_22776": (["3g-PTQ", "4b-EmpiricalStudy"], [], [], "RQ3; RQ4",
        "Benchmarking robustness of quantized LLMs for code generation across model families"),
    "arxiv_2507_02378": (["1a-DataSelection"], [], [], "RQ1",
        "Distribution-consistent and diversity-aware data selection for efficient code LLM training"),
    "arxiv_2508_08332": (["4b-EmpiricalStudy"], [], ["SCOPE-Borderline"], "RQ4",
        "Benchmarking SLM vs LLM for energy-aware code generation; reports both generation and code energy"),
    "arxiv_2512_24570": (["1a-DataSelection", "1b-DataQuality"], [], [], "RQ1",
        "Empirical study of training data optimization techniques for LLM code generation"),
    "doi_10_1109_TCAD_2024_3483089": (["2b-PEFT", "1b-DataQuality"], [], ["SCOPE-Borderline"], "RQ1; RQ2",
        "RTLCoder: data generation + fine-tuning for RTL code generation; borderline Verilog domain"),
    "doi_10_3390_sym17081325": (["2c-Distillation"], [], [], "RQ2",
        "Distilling pseudocode reasoning for lightweight LLM deployment in code generation"),
    "doi_10_48550_arXiv_2501_11006": (["3b-EarlyExit"], [], ["DUPLICATE-2501_11006"], "RQ3",
        "DUPLICATE of 2501_11006 (GREEN-CODE); RL-based early exit for energy-efficient code generation"),
    "doi_10_48550_arXiv_2402_07844": (["4a-Benchmark"], [], [], "RQ4",
        "Mercury: efficiency benchmark for LLM code synthesis"),

    # === 2024 papers ===
    "2401_00788": (["2b-PEFT", "4b-EmpiricalStudy"], [], [], "RQ2; RQ4",
        "Astraios: systematic evaluation of 7 PEFT methods for code LLMs across 4 model scales"),
    "2401_09964": (["3b-EarlyExit"], [], [], "RQ3",
        "Dynamic model inference using intermediate layers for faster code completion"),
    "2402_01035": (["2a-EfficientPretrain"], [], [], "RQ2",
        "Tokenizer design optimization for pre-training efficiency in code LLMs"),
    "2402_14323": (["3e-ContextPruning"], [], [], "RQ3",
        "REPOFUSE: fused dual context with rank truncated generation for efficient repo-level completion"),
    "2403_10059": (["3e-ContextPruning"], [], [], "RQ3",
        "Repoformer: selective retrieval reducing unnecessary RAG overhead for code completion"),
    "2404_16333": (["3l-CodeSpecific"], [], [], "RQ3",
        "SimPy: AI-oriented grammar reducing token usage for LLM code completion"),
    "2405_15189": (["2f-RLTraining"], [], ["SCOPE-GenEffCode"], "RQ2",
        "EffiLearner: LLM self-optimization for generating more runtime-efficient code"),
    "2405_15208": (["3a-SpeculativeDecoding"], [], [], "RQ3",
        "Parallel decoding of lexical units (LUD) for LLM inference speedup including code generation"),
    "2405_15842": (["3i-Routing"], [], [], "RQ3",
        "Model cascading with black-box multi-model framework for cost-efficient code completion"),
    "2406_10263": (["3e-ContextPruning"], [], [], "RQ3",
        "Lightweight adaptive retrieval framework with critique for efficient code completion"),
    "2406_18294": (["3e-ContextPruning"], [], [], "RQ3",
        "Hierarchical context pruning to reduce prompt length for real-world repo-level code completion"),
    "2407_00434": (["1a-DataSelection"], ["3e-ContextPruning"], [], "RQ1",
        "Heuristic file pruning to improve data efficiency for code generation training"),
    "2407_03157": (["3f-KVCache"], [], [], "RQ3",
        "PIE: efficiently updating KV cache for incremental code editing in LLMs"),
    "2407_05040": (["1a-DataSelection", "2b-PEFT"], [], [], "RQ1; RQ2",
        "Data pruning for efficient LLM code generation fine-tuning; fewer data, better alignment"),
    "2407_20042": (["3b-EarlyExit"], [], [], "RQ3",
        "CodeFast: early stopping of excess token generation for faster code completion"),
    "2408_03680": (["2c-Distillation", "2d-CurriculumLearning"], [], [], "RQ2",
        "Self-paced knowledge distillation for lightweight yet effective code LLMs"),
    "2408_11198": (["3n-PromptEng"], [], [], "RQ3",
        "Evolutionary automated prompt engineering for cost-effective code generation"),
    "2409_16416": (["3n-PromptEng"], [], [], "RQ3",
        "Prompt engineering technique selection for code generation via difficulty prediction"),
    "2410_08391": (["3f-KVCache"], [], [], "RQ3",
        "KV Prediction to reduce time-to-first-token for LLM inference including code generation"),
    "2410_10209": (["2f-RLTraining"], [], ["SCOPE-GenEffCode"], "RQ2",
        "EffiCoder: efficiency-aware fine-tuning; focus is improving runtime efficiency of generated code"),
    "2410_14766": (["3g-PTQ", "4b-EmpiricalStudy"], [], [], "RQ3; RQ4",
        "Evaluates quantized LLMs for code generation on low-resource languages"),
    "2410_18194": (["1a-DataSelection"], [], [], "RQ1",
        "ZIP-FIT: compression-based data selection for LM training with code generation experiments"),
    "2410_22480": (["3j-Sampling"], [], [], "RQ3",
        "Optimized sample compute allocation to reduce inference cost for LLM code generation"),
    "2410_22793": (["3d-PromptCompression"], [], [], "RQ3",
        "DocString compression to reduce input tokens and improve code generation efficiency"),
    "2411_05010": (["3j-Sampling"], [], [], "RQ3",
        "Scattered Forest Search: efficient code space exploration reducing iterations needed"),
    "2411_06680": (["3f-KVCache"], [], [], "RQ3",
        "AnchorCoder: anchor attention to reduce KV cache memory for code generation"),
    "2411_11055": (["3a-SpeculativeDecoding"], [], [], "RQ3",
        "FastDraft: efficient draft model training for speculative decoding with code generation"),
    "2412_15441": (["4b-EmpiricalStudy", "3m-SystemOpt"], [], [], "RQ3; RQ4",
        "Empirical evaluation of serving configurations for code SLM resource utilization"),
    "2412_16589": (["2d-CurriculumLearning", "2b-PEFT"], [], [], "RQ2",
        "Context and curriculum-based learning for improving FIM code completion efficiency"),
    "arxiv_2403_00046": (["1a-DataSelection"], [], [], "RQ1",
        "Data-efficient adaptation of LLMs for code generation"),
    "arxiv_2404_02183": (["3k-MultiAgent"], [], [], "RQ3",
        "Self-organized agents for ultra large-scale code generation"),
    "arxiv_2408_06450": (["4b-EmpiricalStudy"], [], [], "RQ4",
        "Evaluation study of language models for efficient code generation"),
    "arxiv_2410_00558": (["2c-Distillation"], [], [], "RQ2",
        "AMR-Evol: adaptive modular response evolution for knowledge distillation in code LLMs"),
    "ss_Zhang2024llm": (["3m-SystemOpt"], [], [], "RQ3",
        "LLM-Cloud Complete: cloud computing system for efficient LLM-based code completion"),

    # === 2023 papers ===
    "2303_05378": (["3g-PTQ"], [], [], "RQ3",
        "Quantization of large code generation models (CodeGen, StarCoder) with quality analysis"),
    "2303_07457": (["3c-NAR"], [], [], "RQ3",
        "AMOM: adaptive masking for non-autoregressive generation including code tasks"),
    "2305_11860": (["3j-Sampling"], [], [], "RQ3",
        "Adaptive-Consistency: adaptive sampling reducing LLM code generation sampling cost"),
    "2308_10462": (["2b-PEFT", "4b-EmpiricalStudy"], [], [], "RQ2; RQ4",
        "Empirical study of PEFT + quantization for code generation with large language models"),
    "2309_14021": (["3h-ModelPruning"], [], [], "RQ3",
        "LORD: low-rank decomposition for one-shot compression of monolingual code LLMs"),
    "2310_18628": (["2c-Distillation"], [], [], "RQ2",
        "Personalised distillation: adaptive learning for efficient code generation from large to small models"),
    "2312_02418": (["1b-DataQuality", "1a-DataSelection"], [], [], "RQ1",
        "Embedding-guided data pruning via synthetic corruptions for code training data quality"),
    "2312_08583": (["3g-PTQ"], [], [], "RQ3",
        "ZeroQuant(4+2): FP6-centric quantization strategy for LLM efficiency"),
    "2312_11511": (["3i-Routing"], [], [], "RQ3",
        "ComplexityNet: task complexity-based routing among LLMs for code generation"),
    "2312_13010": (["3k-MultiAgent"], [], [], "RQ3",
        "AgentCoder: multi-agent code generation with iterative testing reducing token overhead"),
    "Chen2023personalized": (["2c-Distillation"], [], ["DUPLICATE-2310_18628"], "RQ2",
        "DUPLICATE of 2310_18628; Personalised distillation for code LLMs"),
    "arxiv_2304_05216": (["2b-PEFT", "4b-EmpiricalStudy"], [], [], "RQ2; RQ4",
        "Empirical study of efficient fine-tuning methods for pre-trained code models"),
    "arxiv_2312_05562": (["3o-CoTOpt", "2c-Distillation"], [], [], "RQ2; RQ3",
        "Chain-of-Thought for lightweight code LMs; training + inference optimization"),
    "doi_10_1109_ASE56229_2023_00125": (["2b-PEFT", "4b-EmpiricalStudy"], [], [], "RQ2; RQ4",
        "Empirical study of parameter-efficient fine-tuning methods for pre-trained code models"),
    "doi_10_1109_TSE_2023_3267446": (["4a-Benchmark"], [], [], "RQ4",
        "MultiPL-E: scalable polyglot benchmarking framework for neural code generation"),

    # === 2022 paper ===
    "2204_09877": (["3c-NAR"], [], [], "RQ3",
        "Non-autoregressive Transformer for full-line code completion with reported speedup"),

    # === 2020 paper ===
    "arxiv_2004_13651": (["3b-EarlyExit", "3l-CodeSpecific"], [], [], "RQ3",
        "Fast and memory-efficient neural code completion via early exit and code-specific optimizations"),
}


def main():
    # Verify all 123 papers are covered
    import openpyxl
    wb = openpyxl.load_workbook(BASE / "primary-studies-assessment.xlsx")
    ws = wb["Primary Studies"]
    xlsx_keys = set()
    xlsx_data = {}
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, values_only=True):
        key = row[1]
        xlsx_keys.add(key)
        xlsx_data[key] = {
            "num": row[0], "title": row[2], "year": row[4],
            "venue": row[5], "old_lifecycle": row[19],
        }

    classified_keys = set(CLASSIFICATIONS.keys())
    missing = xlsx_keys - classified_keys
    extra = classified_keys - xlsx_keys

    if missing:
        print(f"ERROR: {len(missing)} papers not classified: {missing}")
    if extra:
        print(f"WARNING: {len(extra)} extra keys not in xlsx: {extra}")

    print(f"Total classified: {len(classified_keys)}")
    print(f"Total in xlsx: {len(xlsx_keys)}")
    print(f"Coverage: {len(classified_keys & xlsx_keys)}/{len(xlsx_keys)}")

    # Write CSV
    fieldnames = [
        "num", "key", "title", "year", "venue",
        "old_lifecycle", "new_rq", "primary_categories",
        "secondary_categories", "scope_flags", "brief_rationale",
    ]

    rows = []
    for key in sorted(xlsx_keys, key=lambda k: xlsx_data[k]["num"]):
        d = xlsx_data[key]
        cls = CLASSIFICATIONS.get(key, (["UNCLASSIFIED"], [], [], "UNKNOWN", "Not classified"))
        primary, secondary, flags, rq, rationale = cls
        rows.append({
            "num": d["num"],
            "key": key,
            "title": d["title"],
            "year": d["year"],
            "venue": d["venue"],
            "old_lifecycle": d["old_lifecycle"],
            "new_rq": rq,
            "primary_categories": "; ".join(primary),
            "secondary_categories": "; ".join(secondary),
            "scope_flags": "; ".join(flags),
            "brief_rationale": rationale,
        })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary statistics ──
    from collections import Counter
    rq_counts = Counter()
    cat_counts = Counter()
    scope_counts = Counter()
    duplicates = []

    for key, (primary, secondary, flags, rq, _) in CLASSIFICATIONS.items():
        if key not in xlsx_keys:
            continue
        for rq_part in rq.split(";"):
            rq_counts[rq_part.strip()] += 1
        for cat in primary + secondary:
            cat_counts[cat] += 1
        for flag in flags:
            if flag.startswith("DUPLICATE"):
                duplicates.append(key)
            else:
                scope_counts[flag] += 1

    print("\n=== NEW RQ Distribution ===")
    for rq, count in sorted(rq_counts.items()):
        print(f"  {rq}: {count}")

    print("\n=== Category Distribution (primary + secondary) ===")
    for cat, count in cat_counts.most_common():
        print(f"  {cat}: {count}")

    print("\n=== Scope Flags ===")
    for flag, count in scope_counts.most_common():
        print(f"  {flag}: {count}")

    if duplicates:
        print(f"\n=== Duplicates Found ===")
        for d in duplicates:
            print(f"  {d}: {CLASSIFICATIONS[d][4]}")

    print(f"\nOutput: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
