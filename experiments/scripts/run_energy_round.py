"""能耗补测轮：对 27 个配置各跑 1 次，采集 NVML energy 数据。

用法:
    python run_energy_round.py --device cuda:0 --output-dir results/energy_round

设计：先跑全部 12 个 composition 配置（7B model），再跑全部 15 个 pareto 配置（0.5B-14B）。
与 run1/2/3 数据配合使用，填充 sec-rq6.tex 的 Energy 列。
"""

import argparse
import json
from pathlib import Path

from eval_humaneval import run_benchmark

MODEL_DIR = Path.home() / "efficient-codegen-exp" / "models"
DRAFT_MODEL = str(MODEL_DIR / "Qwen2.5-Coder-0.5B-Instruct")
MAIN_7B = str(MODEL_DIR / "Qwen2.5-Coder-7B-Instruct")

# Composition: 12 configs on 7B
COMPOSITION = [
    ("C01", MAIN_7B, "fp16", "standard",    "greedy"),
    ("C02", MAIN_7B, "int8", "standard",    "greedy"),
    ("C03", MAIN_7B, "int4", "standard",    "greedy"),
    ("C04", MAIN_7B, "fp16", "speculative", "greedy"),
    ("C05", MAIN_7B, "int8", "speculative", "greedy"),
    ("C06", MAIN_7B, "int4", "speculative", "greedy"),
    ("C07", MAIN_7B, "fp16", "standard",    "adaptive"),
    ("C08", MAIN_7B, "int8", "standard",    "adaptive"),
    ("C09", MAIN_7B, "int4", "standard",    "adaptive"),
    ("C10", MAIN_7B, "fp16", "speculative", "adaptive"),
    ("C11", MAIN_7B, "int8", "speculative", "adaptive"),
    ("C12", MAIN_7B, "int4", "speculative", "adaptive"),
]

# Pareto: 15 configs on 5 model scales × 3 precisions
_SIZES = ["0.5B", "1.5B", "3B", "7B", "14B"]
_PRECISIONS = ["fp16", "int8", "int4"]
PARETO = []
_pid = 1
for sz in _SIZES:
    mp = str(MODEL_DIR / f"Qwen2.5-Coder-{sz}-Instruct")
    for prec in _PRECISIONS:
        PARETO.append((f"P{_pid:02d}", mp, prec, "standard", "greedy"))
        _pid += 1


def main():
    parser = argparse.ArgumentParser(description="Energy measurement round")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip", nargs="+", default=[],
                        help="Skip these config IDs (e.g., --skip C01 P05)")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Only run these config IDs")
    args = parser.parse_args()

    base_out = Path(args.output_dir or (Path.home() / "efficient-codegen-exp" / "results" / "energy_round"))
    comp_out = base_out / "composition"
    pareto_out = base_out / "pareto"
    comp_out.mkdir(parents=True, exist_ok=True)
    pareto_out.mkdir(parents=True, exist_ok=True)

    print(f"=== Energy Round ===")
    print(f"Device: {args.device}")
    print(f"Output: {base_out}")
    if args.only:
        print(f"Only: {args.only}")
    if args.skip:
        print(f"Skip: {args.skip}")
    print()

    for cid, model_path, precision, decoding, sampling in COMPOSITION:
        if args.only and cid not in args.only:
            continue
        if cid in args.skip:
            continue
        draft = DRAFT_MODEL if decoding == "speculative" else None
        run_benchmark(
            config_id=cid,
            model_path=model_path,
            precision=precision,
            decoding=decoding,
            sampling=sampling,
            draft_model_path=draft,
            device=args.device,
            output_dir=str(comp_out),
        )

    for cid, model_path, precision, decoding, sampling in PARETO:
        if args.only and cid not in args.only:
            continue
        if cid in args.skip:
            continue
        run_benchmark(
            config_id=cid,
            model_path=model_path,
            precision=precision,
            decoding=decoding,
            sampling=sampling,
            draft_model_path=None,
            device=args.device,
            output_dir=str(pareto_out),
        )

    print("\n=== Energy Round Done ===")


if __name__ == "__main__":
    main()
