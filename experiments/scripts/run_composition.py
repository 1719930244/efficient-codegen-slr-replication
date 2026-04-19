"""实验 1：Technique Composition — 12 配置评测

用法:
    uv run python scripts/run_composition.py                    # 运行全部 12 配置
    uv run python scripts/run_composition.py --configs C01 C04  # 运行指定配置
    uv run python scripts/run_composition.py --device cuda:1    # 指定 GPU
    uv run python scripts/run_composition.py --run 2            # 第 2 轮（共 3 轮）
"""

import argparse
import json
from pathlib import Path

from eval_humaneval import run_benchmark

MODEL_DIR = Path.home() / "efficient-codegen-exp" / "models"
MAIN_MODEL = str(MODEL_DIR / "Qwen2.5-Coder-7B-Instruct")
DRAFT_MODEL = str(MODEL_DIR / "Qwen2.5-Coder-0.5B-Instruct")

# 12 配置矩阵: (config_id, precision, decoding, sampling)
CONFIGS = [
    ("C01", "fp16", "standard",    "greedy"),
    ("C02", "int8", "standard",    "greedy"),
    ("C03", "int4", "standard",    "greedy"),
    ("C04", "fp16", "speculative", "greedy"),
    ("C05", "int8", "speculative", "greedy"),
    ("C06", "int4", "speculative", "greedy"),
    ("C07", "fp16", "standard",    "adaptive"),
    ("C08", "int8", "standard",    "adaptive"),
    ("C09", "int4", "standard",    "adaptive"),
    ("C10", "fp16", "speculative", "adaptive"),
    ("C11", "int8", "speculative", "adaptive"),
    ("C12", "int4", "speculative", "adaptive"),
]


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Technique Composition")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="指定要运行的配置编号，如 C01 C04")
    parser.add_argument("--device", default="cuda:0", help="GPU device")
    parser.add_argument("--run", type=int, default=1, help="第几轮运行 (1-3)")
    parser.add_argument("--output-dir", default=None, help="输出目录")
    args = parser.parse_args()

    output_dir = args.output_dir or str(
        Path.home() / "efficient-codegen-exp" / "results" / "composition" / f"run{args.run}"
    )

    configs = CONFIGS
    if args.configs:
        configs = [(cid, p, d, s) for cid, p, d, s in CONFIGS if cid in args.configs]

    print(f"=== Experiment 1: Technique Composition (Run {args.run}) ===")
    print(f"Configs to run: {[c[0] for c in configs]}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print()

    results = []
    for config_id, precision, decoding, sampling in configs:
        draft = DRAFT_MODEL if decoding == "speculative" else None
        result = run_benchmark(
            config_id=config_id,
            model_path=MAIN_MODEL,
            precision=precision,
            decoding=decoding,
            sampling=sampling,
            draft_model_path=draft,
            device=args.device,
            output_dir=output_dir,
        )
        results.append(result)

    # 汇总表
    print(f"\n{'='*80}")
    print(f"{'Config':>6} | {'Precision':>9} | {'Decoding':>12} | {'Sampling':>10} | "
          f"{'pass@1':>7} | {'Latency':>9} | {'Tok/s':>7} | {'Memory':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r.config_id:>6} | {r.precision:>9} | {r.decoding:>12} | {r.sampling:>10} | "
              f"{r.pass_at_1:>6.1f}% | {r.mean_total_time_ms:>7.0f}ms | "
              f"{r.mean_tokens_per_sec:>7.1f} | {r.peak_gpu_memory_mb:>6.0f}MB")

    # 保存汇总
    summary_path = Path(output_dir) / "summary.json"
    from dataclasses import asdict
    with open(summary_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
