"""MBPP 基准评测（完全复用 eval_humaneval 的模型加载/生成/计时/结果 JSON 结构）

数据: JSONL 文件（--data 指定），每行一个题目，含
    task_id / text / code / test_list / test_setup_code 等字段。
    数据划分为 full 974 题（文献中另有 sanitized-427 / MBPP+ 等划分，
    数字互不可比），需在论文中注明。
Prompt: 由任务描述 (text) + 前三个示例测试 (test_list[:3]，与 MBPP 原始论文 /
    EvalPlus / bigcode-eval 的主流做法一致) + 函数签名（用 AST 在参考 code 中
    定位被测函数的 def，支持跨行签名）组成，模型生成函数体。
判题: 子进程执行 函数签名 + 生成代码 + test_list 全部断言（带超时与 RLIMIT
    资源限制）。

用法:
    uv run python scripts/eval_mbpp.py                        # 运行全部 12 配置
    uv run python scripts/eval_mbpp.py --configs C01 C04      # 运行指定配置
    uv run python scripts/eval_mbpp.py --device cuda:1        # 指定 GPU
    uv run python scripts/eval_mbpp.py --run 2                # 第 2 轮
    uv run python scripts/eval_mbpp.py --data /path/to/mbpp.jsonl
"""

import argparse
import ast
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

DEFAULT_DATA = str(Path.home() / "efficient-codegen-exp" / "data" / "mbpp.jsonl")

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

def _tested_function_name(problem: dict) -> Optional[str]:
    """从 test_list 解析被测函数名（assert 语句中直接调用的函数名）"""
    for t in problem.get("test_list") or []:
        try:
            tree = ast.parse(t)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                return node.func.id
    return None


def extract_signature(problem: dict) -> str:
    """用 AST 在参考 code 中定位被测函数的 def 并提取完整签名

    修复 [MBPP-高]: 旧实现取参考 code 的第一个非空行当签名，但实测 199/974 题
    （约 20.4%）首行是 import/常量而非 def（如 task_id 1 → 'R = 3'、
    task_id 3 → 'import math'、task_id 18 → 'NO_OF_CHARS = 256'），导致这些题
    的 prompt 不含函数签名/参数、判题文件前置错误的语句，pass@1 相对官方口径
    系统性偏低。
    修复 [MBPP-中]: 支持括号未闭合的跨行签名（如 task_id 702
    'def find_ind(key, i, n,\\n k, arr):'）——签名取 def 行到函数体第一条语句
    之前的所有行。另外优先按 test_list 中被调用的函数名定位 def，避免选中
    先定义的辅助函数（如 task_id 702 的 find_ind 是辅助函数，被测的是
    removals）。
    """
    code = (problem.get("code") or "").strip()
    target = _tested_function_name(problem)
    if code:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            tree = None
        if tree is not None:
            funcs = sorted(
                (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))),
                key=lambda n: n.lineno,
            )
            top_funcs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            node = None
            if target:
                node = next((f for f in top_funcs if f.name == target), None) \
                    or next((f for f in funcs if f.name == target), None)
            if node is None:
                node = top_funcs[0] if top_funcs else (funcs[0] if funcs else None)
            if node is not None and node.body:
                lines = code.splitlines()
                start = node.lineno - 1        # def 行（1-based lineno 转 0-based）
                end = node.body[0].lineno - 1  # 签名止于函数体第一条语句之前
                sig_lines = [l.rstrip() for l in lines[start:max(end, start + 1)]]
                if sig_lines:
                    return "\n".join(sig_lines)
    # 兜底：按被测函数名拼通用签名（与原实现一致；实测数据中空 code 为 0）
    name = target or f"task_{problem.get('task_id', 0)}"
    return f"def {name}(*args, **kwargs):"


def build_prompt(problem: dict) -> str:
    """构造 prompt: 任务描述 + 前三个示例测试 + 函数签名（模型续写函数体）

    修复 [MBPP-中]: 旧实现只放 test_list[0] 一个示例测试；主流做法（MBPP 原始
    论文、EvalPlus、bigcode-eval）是放 test_list[:3] 三个示例。示例信息量不同
    会直接改变通过率，与已发表数字不可比。
    """
    text = (problem.get("text") or "").strip()
    tests = problem.get("test_list") or []
    examples = "\n".join(t.strip() for t in tests[:3])
    doc = f"{text}\n{examples}" if examples else text
    return f'"""\n{doc}\n"""\n{extract_signature(problem)}\n'


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


def _make_preexec(timeout_s: int):
    """构造判题子进程的资源限制函数（仅 POSIX；与共享 GPU 机器上其他进程共存）

    判题子进程无沙箱，测试代码可能失控；这里只设 RLIMIT（不设
    RLIMIT_NPROC——它按用户计数，会波及同用户的长跑实验）。
    """
    if os.name != "posix":
        return None
    try:
        import resource
    except ImportError:  # Windows 等无 resource 模块的平台
        return None

    def _preexec():
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


def check_correctness(problem: dict, completion: str, timeout: int = 5) -> tuple[bool, Optional[str]]:
    """子进程执行生成代码 + test_list，返回 (passed, fail_reason)，通过时
    fail_reason 为 None（子进程带超时与 RLIMIT 资源限制）"""
    sig = extract_signature(problem)
    tests = problem.get("test_list") or []
    test_code = sig + "\n" + completion.rstrip("\n") + "\n\n"
    setup = (problem.get("test_setup_code") or "").strip()
    if setup:
        test_code += setup + "\n\n"
    test_code += "\n".join(tests) + "\n"

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


# ── 自适应采样（与 eval_humaneval.adaptive_sampling 同逻辑，使用 MBPP 判题）──

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
    timeout: int = 5,
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
    timeout: int = 5,
) -> BenchmarkResult:
    """运行一个完整的 MBPP 评测配置（流程与 eval_humaneval.run_benchmark 一致）"""

    print(f"\n{'='*60}")
    print(f"Config: {config_id}  [benchmark=mbpp]")
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
    print(f"Loaded {len(problems)} MBPP problems from {data_path}")

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
    parser = argparse.ArgumentParser(description="MBPP Benchmark (same config matrix as Experiment 1)")
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help=f"MBPP JSONL 数据路径 (默认 {DEFAULT_DATA})")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="指定要运行的配置编号，如 C01 C04")
    parser.add_argument("--device", default="cuda:0", help="GPU device")
    parser.add_argument("--run", type=int, default=1, help="第几轮运行 (1-3)")
    parser.add_argument("--output-dir", default=None, help="输出目录")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="最大生成 token 数")
    parser.add_argument("--timeout", type=int, default=5, help="判题子进程超时(秒)")
    args = parser.parse_args()

    data_path = str(Path(args.data).expanduser())
    if not Path(data_path).exists():
        raise SystemExit(f"[ERROR] 数据文件不存在: {data_path} (用 --data 指定)")

    output_dir = args.output_dir or str(
        Path.home() / "efficient-codegen-exp" / "results" / "mbpp" / f"run{args.run}"
    )

    configs = CONFIGS
    if args.configs:
        configs = [(cid, p, d, s) for cid, p, d, s in CONFIGS if cid in args.configs]

    print(f"=== MBPP Benchmark (Run {args.run}) ===")
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
