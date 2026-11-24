"""实验 2：Pareto Frontier — 5 模型 × 3 精度 = 15 配置

用法:
    uv run python scripts/run_pareto.py                          # 运行全部 15 配置
    uv run python scripts/run_pareto.py --models 0.5B 7B         # 指定模型尺寸
    uv run python scripts/run_pareto.py --precisions fp16 int4   # 指定精度
    uv run python scripts/run_pareto.py --device cuda:2          # 指定 GPU
    uv run python scripts/run_pareto.py --run 2                  # 第 2 轮
"""

import argparse
import json
from pathlib import Path

from eval_humaneval import run_benchmark

MODEL_DIR = Path.home() / "efficient-codegen-exp" / "models"

# 模型列表: (size_label, dir_name)
MODELS = [
    ("0.5B", "Qwen2.5-Coder-0.5B-Instruct"),
    ("1.5B", "Qwen2.5-Coder-1.5B-Instruct"),
    ("3B",   "Qwen2.5-Coder-3B-Instruct"),
    ("7B",   "Qwen2.5-Coder-7B-Instruct"),
    ("14B",  "Qwen2.5-Coder-14B-Instruct"),
]

PRECISIONS = ["fp16", "int8", "int4"]


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Pareto Frontier")
    parser.add_argument("--models", nargs="+", default=None,
                        help="指定模型尺寸，如 0.5B 7B 14B")
    parser.add_argument("--precisions", nargs="+", default=None,
                        help="指定精度，如 fp16 int4")
    parser.add_argument("--device", default="cuda:0", help="GPU device")
    parser.add_argument("--run", type=int, default=1, help="第几轮运行 (1-3)")
    parser.add_argument("--output-dir", default=None, help="输出目录")
    args = parser.parse_args()

    output_dir = args.output_dir or str(
        Path.home() / "efficient-codegen-exp" / "results" / "pareto" / f"run{args.run}"
    )

    models = MODELS
    if args.models:
        models = [(s, d) for s, d in MODELS if s in args.models]

    precisions = args.precisions or PRECISIONS

    configs = []
    idx = 1
    for size, dirname in models:
        for prec in precisions:
            config_id = f"P{idx:02d}"
            configs.append((config_id, size, dirname, prec))
            idx += 1

    print(f"=== Experiment 2: Pareto Frontier (Run {args.run}) ===")
    print(f"Models: {[m[0] for m in models]}")
    print(f"Precisions: {precisions}")
    print(f"Total configs: {len(configs)}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print()

    results = []
    for config_id, size, dirname, precision in configs:
        model_path = str(MODEL_DIR / dirname)
        result = run_benchmark(
            config_id=config_id,
            model_path=model_path,
            precision=precision,
            decoding="standard",
            sampling="greedy",
            device=args.device,
            output_dir=output_dir,
        )
        results.append(result)

    # 汇总表
    print(f"\n{'='*80}")
    print(f"{'Config':>6} | {'Model':>35} | {'Precision':>9} | "
          f"{'pass@1':>7} | {'Latency':>9} | {'Tok/s':>7} | {'Memory':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r.config_id:>6} | {r.model_name:>35} | {r.precision:>9} | "
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
