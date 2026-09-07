"""BigCodeBench 基准评测（完全复用 eval_humaneval 的模型加载/生成/计时/结果 JSON 结构）

数据: JSONL 文件（--data 指定），默认 bigcodebench_full.jsonl（completion 划分，
    full 1140 题；官方另有面向 instruct 模型的 160 题子集）。每行一个题目，含
    task_id / complete_prompt / instruct_prompt / test / entry_point / libs 等字段。
Prompt: 直接使用题目的 complete_prompt 字段（imports + def task_func 签名 +
    docstring，合法 Python 前缀），模型续写函数体。注意 instruct_prompt 是自然
    语言描述，不是合法 Python，不能用作生成前缀/判题文件头部。
    方法学说明: 与 HumanEval/MBPP 保持一致，三个基准统一用裸文本续写（不用
    chat template 包裹 instruct_prompt），偏离 BigCodeBench 官方 instruct 做法。
判题: 子进程执行 complete_prompt + 生成代码 + test + unittest 运行器（带超时与
    RLIMIT 资源限制）；测试数据全部为 unittest.TestCase 类（无 HumanEval 风格的
    check(...) 调用、无自带运行器），因此拼接后必须追加运行器并以
    wasSuccessful() 决定退出码，否则任何可 exec 的补全都会假通过。
    依赖缺失记为 fail，并在结果 details[*].fail_reason 中注明
    missing_dependency:<模块名>。
超时: 固定 --timeout 秒（默认 20s），为官方"按 ground-truth 运行时长自适应
    (min 1s, 4×GT, 有上限)"的近似，需在论文中注明。

用法:
    uv run python scripts/eval_bigcodebench.py                        # 运行全部 12 配置
    uv run python scripts/eval_bigcodebench.py --configs C01 C04      # 运行指定配置
    uv run python scripts/eval_bigcodebench.py --device cuda:1        # 指定 GPU
    uv run python scripts/eval_bigcodebench.py --run 2                # 第 2 轮
    uv run python scripts/eval_bigcodebench.py --data /path/to/bigcodebench.jsonl
"""

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Optional

# 保证从 experiments/ 目录运行时也能导入同目录的 eval_humaneval
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from eval_humaneval import (
    BenchmarkResult,
    GenerationResult,
    generate_completion,
    get_peak_memory_mb,
    load_model,
    pass_at_k,
)

MODEL_DIR = Path.home() / "efficient-codegen-exp" / "models"
MAIN_MODEL = str(MODEL_DIR / "Qwen2.5-Coder-7B-Instruct")
DRAFT_MODEL = str(MODEL_DIR / "Qwen2.5-Coder-0.5B-Instruct")

# 修复 [BCB-致命4]: 原默认路径 bigcodebench.jsonl 不存在，实际数据文件为
# bigcodebench_full.jsonl（completion 划分）与 bigcodebench_instruct.jsonl，
# 二者仅 prompt 字段不同，instruct_prompt/complete_prompt 字段完全一致。
DEFAULT_DATA = str(Path.home() / "efficient-codegen-exp" / "data" / "bigcodebench_full.jsonl")

# 与 run_composition.py 完全一致的 12 配置矩阵: (config_id, precision, decoding, sampling)
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

_MODULE_NOT_FOUND_RE = re.compile(r"ModuleNotFoundError: No module named '([^']+)'")
_IMPORT_ERROR_RE = re.compile(r"ImportError: cannot import name '[^']*' from '([^']+)'")


# ── 数据加载 ──────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    """加载 JSONL 数据集"""
    problems = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


# ── Prompt 构造 ───────────────────────────────────────────

def build_prompt(problem: dict) -> str:
    """直接使用题目的 complete_prompt 作为 prompt（completion 模式）

    修复 [BCB-致命1]: instruct_prompt 是自然语言（'Calculates the average of...'
    开头），不是合法 Python，既不能作为补全前缀，也不能拼进判题文件头部
    （会导致全部题目 SyntaxError、pass@1 恒为 0）。complete_prompt =
    imports + def task_func 签名 + docstring，是合法 Python 前缀。
    """
    prompt = problem.get("complete_prompt") or ""
    if not prompt.endswith("\n"):
        prompt += "\n"
    return prompt


# ── 代码执行评测 ──────────────────────────────────────────

def _missing_module(stderr: str) -> Optional[str]:
    """从子进程 stderr 中提取缺失的依赖模块名，无则返回 None"""
    m = _MODULE_NOT_FOUND_RE.search(stderr or "")
    if m:
        return m.group(1).split(".")[0]
    m = _IMPORT_ERROR_RE.search(stderr or "")
    if m:
        return m.group(1).split(".")[0]
    return None


# 修复 [BCB-致命3]: 数据中 0/1140 题的 test 字段自带运行器调用（测试全是
# unittest.TestCase 类），若只定义不运行，任何能被 exec 的补全（哪怕逻辑全错）
# 都会 returncode=0 假通过。按官方 harness 做法，拼接后追加 unittest 运行器，
# 以 wasSuccessful() 决定退出码。
# 注意 [BCB-致命2]: 不能追加 HumanEval 风格的 check(entry_point) —— 实测 0/1140
# 题存在 check 函数定义/调用，追加行必然 NameError，所有题判失败。
_UNITTEST_RUNNER = (
    "\n\nif __name__ == '__main__':\n"
    "    import sys\n"
    "    import unittest\n"
    "    _unittest_result = unittest.main(exit=False).result\n"
    "    sys.exit(0 if _unittest_result.wasSuccessful() else 1)\n"
)


def _make_preexec(timeout_s: int):
    """构造判题子进程的资源限制函数（仅 POSIX；与共享 GPU 机器上其他进程共存）

    BCB 测试含 turtle/socket 等副作用代码，无沙箱时可能失控；这里只设
    RLIMIT（不设 RLIMIT_NPROC——它是按用户计数，会波及同用户的长跑实验）。
    """
    if os.name != "posix":
        return None
    try:
        import resource
    except ImportError:  # Windows 等无 resource 模块的平台
        return None

    def _preexec():
        # CPU 时间上限给足多线程导入（如 tensorflow）的余量；墙钟超时仍由
        # subprocess timeout 控制。地址空间 16 GiB 足够覆盖测试内的常规计算。
        rlimits = [
            (resource.RLIMIT_CPU, max(int(timeout_s) * 8, 120)),
            (resource.RLIMIT_AS, 16 * 1024 ** 3),
            (resource.RLIMIT_FSIZE, 256 * 1024 ** 2),
            (resource.RLIMIT_NOFILE, 1024),
        ]
        for res, val in rlimits:
            try:
                resource.setrlimit(res, (val, val))
            except (ValueError, OSError):
                pass

    return _preexec


def check_correctness(problem: dict, completion: str, timeout: int = 20) -> tuple[bool, Optional[str]]:
    """子进程执行 complete_prompt + 生成代码 + test + unittest 运行器，
    返回 (passed, fail_reason)

    通过时 fail_reason 为 None；依赖缺失记为 fail 并注明
    missing_dependency:<模块名>。
    """
    test = (problem.get("test") or "").strip("\n")
    test_code = (
        build_prompt(problem)
        + completion.rstrip("\n") + "\n\n"
        + test
        + _UNITTEST_RUNNER
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(test_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            timeout=timeout,
            text=True,
            preexec_fn=_make_preexec(timeout),
        )
        if result.returncode == 0:
            return True, None
        mod = _missing_module(result.stderr)
        if mod:
            return False, f"missing_dependency:{mod}"
        return False, "runtime_error"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        print(f"    [WARN] check_correctness infrastructure error: {type(e).__name__}: {e}")
        return False, f"infra_error:{type(e).__name__}"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── 自适应采样（与 eval_humaneval.adaptive_sampling 同逻辑，使用 BigCodeBench 判题）──

def adaptive_sampling(
    model,
    tokenizer,
    prompt: str,
    problem: dict,
    n: int = 10,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    confidence_threshold: float = 0.95,
    assistant_model=None,
    assistant_tokenizer=None,
    device: str = "cuda:0",
    timeout: int = 20,
) -> tuple[list[str], list[bool], list[str], float, float, int, float]:
    """自适应采样：达到置信度阈值即早停

    返回 (completions, passed_list, fail_reasons, mean_ms_per_token,
          total_time_ms, total_tokens, total_energy_mj)
    """
    completions = []
    passed_list = []
    fail_reasons = []
    total_time_ms = 0
    total_tokens = 0
    total_energy_mj = 0.0
    any_energy_measured = False
    mspt_list = []

    for i in range(n):
        comp, mspt, t_ms, n_tok, e_mj = generate_completion(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            assistant_model=assistant_model,
            assistant_tokenizer=assistant_tokenizer,
            device=device,
        )
        completions.append(comp)
        passed, reason = check_correctness(problem, comp, timeout)
        passed_list.append(passed)
        fail_reasons.append(reason if reason else "passed")
        total_time_ms += t_ms
        total_tokens += n_tok
        if e_mj >= 0:
            total_energy_mj += e_mj
            any_energy_measured = True
        mspt_list.append(mspt)

        # 置信度早停
        pass_rate = sum(passed_list) / len(passed_list)
        if len(passed_list) >= 3 and pass_rate >= confidence_threshold:
            break

    mean_mspt = float(np.mean(mspt_list)) if mspt_list else 0.0
    agg_energy_mj = total_energy_mj if any_energy_measured else -1.0
    return completions, passed_list, fail_reasons, mean_mspt, total_time_ms, total_tokens, agg_energy_mj


# ── 主评测流程 ──────────────────────────────────────────────

def run_benchmark(
    config_id: str,
    model_path: str,
    precision: str = "fp16",
    decoding: str = "standard",
    sampling: str = "greedy",
    draft_model_path: Optional[str] = None,
    device: str = "cuda:0",
    output_dir: str = "results",
    data_path: str = DEFAULT_DATA,
    max_new_tokens: int = 512,
    timeout: int = 20,
) -> BenchmarkResult:
    """运行一个完整的 BigCodeBench 评测配置（流程与 eval_humaneval.run_benchmark 一致）"""

    print(f"\n{'='*60}")
    print(f"Config: {config_id}  [benchmark=bigcodebench]")
    print(f"  Model: {model_path}")
    print(f"  Precision: {precision}, Decoding: {decoding}, Sampling: {sampling}")
    print(f"{'='*60}")

    if sampling not in ("greedy", "adaptive"):
        raise ValueError(f"Unknown sampling: {sampling}")

    # 重置显存统计
    torch.cuda.reset_peak_memory_stats()

    # 加载模型
    print("Loading model...")
    model, tokenizer = load_model(model_path, precision, device)

    # 草稿模型始终用 FP16，不跟随主模型精度
    assistant_model = None
    assistant_tokenizer = None
    if decoding == "speculative" and draft_model_path:
        print(f"Loading draft model (FP16): {draft_model_path}")
        assistant_model, assistant_tokenizer = load_model(draft_model_path, "fp16", device)

    # 加载数据
    problems = load_jsonl(data_path)
    print(f"Loaded {len(problems)} BigCodeBench problems from {data_path}")
    print("注意: 依赖缺失的题目将记为 fail (fail_reason=missing_dependency:<模块名>)")

    results = []
    details = []
    all_times = []

    for i, problem in enumerate(problems):
        task_id = str(problem.get("task_id", i))
        prompt = build_prompt(problem)

        if sampling == "greedy":
            comp, mspt, total_ms, n_tok, e_mj = generate_completion(
                model, tokenizer, prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                assistant_model=assistant_model,
                assistant_tokenizer=assistant_tokenizer,
                device=device,
            )
            passed, reason = check_correctness(problem, comp, timeout)

            gr = GenerationResult(
                task_id=task_id, prompt=prompt, completion=comp,
                passed=passed, ms_per_token=mspt, total_time_ms=total_ms,
                tokens_generated=n_tok,
                tokens_per_sec=n_tok / (total_ms / 1000) if total_ms > 0 else 0,
                energy_mj=e_mj,
            )
            results.append(gr)
            all_times.append(total_ms)
            details.append({**asdict(gr), "fail_reason": reason})

        elif sampling == "adaptive":
            comps, passed_list, fail_reasons, mean_mspt, total_ms, total_tokens, e_mj = adaptive_sampling(
                model, tokenizer, prompt, problem,
                n=10, temperature=0.8,
                max_new_tokens=max_new_tokens,
                assistant_model=assistant_model,
                assistant_tokenizer=assistant_tokenizer,
                device=device,
                timeout=timeout,
            )
            best_idx = next((j for j, p in enumerate(passed_list) if p), 0)
            if any(passed_list):
                reason = None
            else:
                cnt = Counter(r for r in fail_reasons if r != "passed")
                reason = "; ".join(f"{k}x{v}" for k, v in sorted(cnt.items())) or "unknown"

            gr = GenerationResult(
                task_id=task_id, prompt=prompt, completion=comps[best_idx],
                passed=any(passed_list), ms_per_token=mean_mspt, total_time_ms=total_ms,
                tokens_generated=total_tokens,
                tokens_per_sec=total_tokens / (total_ms / 1000) if total_ms > 0 else 0,
                energy_mj=e_mj,
            )
            results.append(gr)
            all_times.append(total_ms)
            details.append({**asdict(gr), "fail_reason": reason})

        status = "PASS" if results[-1].passed else "FAIL"
        extra = f" [{details[-1]['fail_reason']}]" if details[-1]["fail_reason"] else ""
        print(f"  [{i+1:3d}/{len(problems)}] {task_id}: {status}{extra} "
              f"({results[-1].total_time_ms:.0f}ms, {results[-1].tokens_generated} tok)")

    # 汇总（与 eval_humaneval 相同的聚合逻辑）
    peak_mem = get_peak_memory_mb(device)
    times = np.array(all_times)
    pass_count = sum(1 for r in results if r.passed)
    n_problems = len(results)

    # pass@10（仅对 adaptive 有意义）
    p_at_10 = 0.0
    if sampling == "adaptive" and n_problems > 0:
        p_at_10 = pass_at_k(10, pass_count, 10) * 100

    # Energy aggregation (skip -1 sentinels from any unsupported measurement)
    valid_energies = [r.energy_mj for r in results if r.energy_mj >= 0]
    if valid_energies:
        total_energy_j_val = float(sum(valid_energies) / 1000.0)
        mean_energy_j_per_req_val = float(np.mean(valid_energies) / 1000.0)
        total_tokens_for_e = sum(r.tokens_generated for r in results if r.energy_mj >= 0)
        mean_energy_j_per_token_val = float(
            sum(valid_energies) / 1000.0 / max(total_tokens_for_e, 1)
        )
    else:
        total_energy_j_val = -1.0
        mean_energy_j_per_req_val = -1.0
        mean_energy_j_per_token_val = -1.0

    benchmark = BenchmarkResult(
        config_id=config_id,
        model_name=Path(model_path).name,
        precision=precision,
        decoding=decoding,
        sampling=sampling,
        pass_at_1=pass_count / n_problems * 100 if n_problems > 0 else 0,
        pass_at_10=p_at_10,
        mean_ms_per_token=float(np.mean([r.ms_per_token for r in results])),
        mean_total_time_ms=float(np.mean(times)),
        p50_total_time_ms=float(np.percentile(times, 50)),
        p95_total_time_ms=float(np.percentile(times, 95)),
        mean_tokens_per_sec=float(np.mean([r.tokens_per_sec for r in results])),
        peak_gpu_memory_mb=peak_mem,
        total_tokens_generated=sum(r.tokens_generated for r in results),
        total_energy_j=total_energy_j_val,
        mean_energy_j_per_request=mean_energy_j_per_req_val,
        mean_energy_j_per_token=mean_energy_j_per_token_val,
        num_problems=n_problems,
        details=details,
    )

    # 保存结果（与 HumanEval 相同的 {config_id}.json 结构）
    out_path = Path(output_dir) / f"{config_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(asdict(benchmark), f, indent=2, ensure_ascii=False)
    print(f"\n  pass@1: {benchmark.pass_at_1:.1f}%")
    print(f"  Mean latency: {benchmark.mean_total_time_ms:.0f}ms")
    print(f"  ms/token: {benchmark.mean_ms_per_token:.1f}")
    print(f"  Tokens/sec: {benchmark.mean_tokens_per_sec:.1f}")
    print(f"  Peak GPU memory: {benchmark.peak_gpu_memory_mb:.0f} MB")
    if benchmark.mean_energy_j_per_request >= 0:
        print(f"  Energy/req: {benchmark.mean_energy_j_per_request:.2f} J, "
              f"Energy/tok: {benchmark.mean_energy_j_per_token*1000:.2f} mJ")
    else:
        print(f"  Energy: not measured")
    reason_counts = Counter(
        d["fail_reason"].split(":")[0] for d in details if d.get("fail_reason")
    )
    if reason_counts:
        print(f"  Fail reasons: {dict(reason_counts)}")
    print(f"  Saved to {out_path}")

    # 正确清理显存，防止 OOM
    del model
    if assistant_model is not None:
        del assistant_model
    if assistant_tokenizer is not None:
        del assistant_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return benchmark


def main():
    parser = argparse.ArgumentParser(description="BigCodeBench Benchmark (same config matrix as Experiment 1)")
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help=f"BigCodeBench JSONL 数据路径 (默认 {DEFAULT_DATA})")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="指定要运行的配置编号，如 C01 C04")
    parser.add_argument("--device", default="cuda:0", help="GPU device")
    parser.add_argument("--run", type=int, default=1, help="第几轮运行 (1-3)")
    parser.add_argument("--output-dir", default=None, help="输出目录")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="最大生成 token 数")
    parser.add_argument("--timeout", type=int, default=20, help="判题子进程超时(秒)")
    args = parser.parse_args()

    data_path = str(Path(args.data).expanduser())
    if not Path(data_path).exists():
        raise SystemExit(f"[ERROR] 数据文件不存在: {data_path} (用 --data 指定)")

    output_dir = args.output_dir or str(
        Path.home() / "efficient-codegen-exp" / "results" / "bigcodebench" / f"run{args.run}"
    )

    configs = CONFIGS
    if args.configs:
        configs = [(cid, p, d, s) for cid, p, d, s in CONFIGS if cid in args.configs]

    print(f"=== BigCodeBench Benchmark (Run {args.run}) ===")
    print(f"Configs to run: {[c[0] for c in configs]}")
    print(f"Data: {data_path}")
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
            data_path=data_path,
            max_new_tokens=args.max_new_tokens,
            timeout=args.timeout,
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
    with open(summary_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
