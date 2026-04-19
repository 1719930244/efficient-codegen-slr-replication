"""HumanEval 评测核心模块

提供模型加载（FP16/INT8/INT4）、代码生成、执行评测功能。
"""

import gc
import json
import time
import os
import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# [FIX #3] HuggingFace 不可达，使用 hf-mirror
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch

# [FIX #21] Monkey-patch transformers 5.5.4 AssistantToTargetTranslator.__init__
# to force `assistant_prune_lm_head = False` when the prune attrs weren't created.
# When main and draft tokenizers share vocab (Qwen-Coder 0.5B + 7B), the init
# block guarded by `len(self._suppress_input_ids) > 0` does not run, so
# `map_input_embeddings` and `assistant_overlap_token_ids` are never set. But
# `self.assistant_prune_lm_head` stays True, and three downstream methods
# (`unmap_input_ids`, `get_target_ids`, `get_target_logits`) dereference the
# missing attrs via that flag. Forcing the flag to False routes all gated
# branches to the safe else-path.
try:
    from transformers.generation.candidate_generator import AssistantToTargetTranslator as _ATM
    _orig_init = _ATM.__init__
    def _guarded_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        if getattr(self, "assistant_prune_lm_head", False) and not hasattr(self, "map_input_embeddings"):
            self.assistant_prune_lm_head = False
    _ATM.__init__ = _guarded_init
except Exception as _e:
    import warnings as _w
    _w.warn(f"FIX #21 monkey-patch failed: {_e}")

import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# -- NVML Energy Measurement --
_NVML_INITED = False
_NVML_HANDLES = {}
_NVML_AVAILABLE = True

def _nvml_handle(device):
    """Return NVML handle for given device string like cuda:0, or None if unavailable."""
    global _NVML_INITED, _NVML_AVAILABLE
    if not _NVML_AVAILABLE:
        return None
    try:
        import pynvml
        if not _NVML_INITED:
            pynvml.nvmlInit()
            _NVML_INITED = True
        idx = int(device.split(":")[-1]) if ":" in device else 0
        if idx not in _NVML_HANDLES:
            _NVML_HANDLES[idx] = pynvml.nvmlDeviceGetHandleByIndex(idx)
        return _NVML_HANDLES[idx]
    except Exception as e:
        print(f"[WARN] NVML init failed, energy measurement disabled: {e}")
        _NVML_AVAILABLE = False
        return None


def read_energy_mj(device):
    """Read NVML total energy counter in millijoules. Returns -1 if unsupported."""
    h = _nvml_handle(device)
    if h is None:
        return -1
    try:
        import pynvml
        return int(pynvml.nvmlDeviceGetTotalEnergyConsumption(h))
    except Exception:
        return -1



# ── 数据类 ──────────────────────────────────────────────

@dataclass
class GenerationResult:
    """单个问题的生成结果"""
    task_id: str
    prompt: str
    completion: str
    passed: bool = False
    ms_per_token: float = 0.0  # [FIX #8] 改名，不再伪装为 TTFT
    total_time_ms: float = 0.0
    tokens_generated: int = 0
    tokens_per_sec: float = 0.0
    energy_mj: float = -1.0  # NVML total energy (mJ), -1 if unsupported


@dataclass
class BenchmarkResult:
    """单个配置的评测结果汇总"""
    config_id: str
    model_name: str
    precision: str
    decoding: str
    sampling: str
    pass_at_1: float = 0.0
    pass_at_10: float = 0.0
    mean_ms_per_token: float = 0.0  # [FIX #8]
    mean_total_time_ms: float = 0.0
    p50_total_time_ms: float = 0.0
    p95_total_time_ms: float = 0.0
    mean_tokens_per_sec: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    total_tokens_generated: int = 0
    total_energy_j: float = -1.0
    mean_energy_j_per_request: float = -1.0
    mean_energy_j_per_token: float = -1.0
    num_problems: int = 0
    details: list = field(default_factory=list)


# ── 模型加载 ──────────────────────────────────────────────

def load_model(model_path: str, precision: str = "fp16", device: str = "cuda:0"):
    """加载模型，支持 FP16/INT8/INT4

    返回 (model, tokenizer, actual_device)
    actual_device: 模型实际所在设备，量化模型为 "auto"
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "trust_remote_code": True,
    }

    if precision == "fp16":
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["device_map"] = device
    elif precision == "int8":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["device_map"] = {"": device}  # [FIX #1] 强制到指定 GPU
    elif precision == "int4":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["device_map"] = {"": device}  # [FIX #1] 强制到指定 GPU
    else:
        raise ValueError(f"Unknown precision: {precision}")

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()
    return model, tokenizer


def get_peak_memory_mb(device: str = "cuda:0") -> float:
    """获取峰值 GPU 显存"""
    # [FIX #7] 如果使用多 GPU，汇总所有设备
    if not device.startswith("cuda"):
        return sum(
            torch.cuda.max_memory_allocated(i) / 1024 / 1024
            for i in range(torch.cuda.device_count())
        )
    idx = int(device.split(":")[-1]) if ":" in device else 0
    return torch.cuda.max_memory_allocated(idx) / 1024 / 1024


# ── HumanEval 数据加载 ──────────────────────────────────

def load_humaneval():
    """加载 HumanEval 数据集（通过 hf-mirror）"""
    ds = load_dataset("openai/openai_humaneval", split="test")
    return list(ds)


# ── 代码生成 ──────────────────────────────────────────────

def generate_completion(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    assistant_model=None,
    assistant_tokenizer=None,
    device: str = "cuda:0",
) -> tuple[str, float, float, int]:
    """生成代码补全，返回 (completion, ms_per_token, total_time_ms, num_tokens)"""

    # [FIX #1] 使用模型实际所在设备
    input_device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(input_device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
    if assistant_model is not None:
        gen_kwargs["assistant_model"] = assistant_model
        # [FIX #18] transformers 5.x requires `tokenizer` and `assistant_tokenizer`
        # for speculative decoding even when they share vocab; otherwise it raises
        # "main and assistant models have different tokenizers".
        gen_kwargs["tokenizer"] = tokenizer
        if assistant_tokenizer is not None:
            gen_kwargs["assistant_tokenizer"] = assistant_tokenizer

    torch.cuda.synchronize()
    e0 = read_energy_mj(device)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start
    e1 = read_energy_mj(device)
    energy_mj = float(e1 - e0) if (e0 >= 0 and e1 >= 0) else -1.0

    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]
    num_tokens = len(new_tokens)
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

    completion = truncate_completion(completion)

    total_time_ms = total_time * 1000
    ms_per_token = total_time_ms / max(num_tokens, 1)

    return completion, ms_per_token, total_time_ms, num_tokens, energy_mj


def truncate_completion(completion: str) -> str:
    """截断补全到第一个完整函数体"""
    lines = completion.split("\n")
    result = []
    for line in lines:
        # [FIX #6] 遇到任何非缩进、非空、非注释、非装饰器的顶层代码即停止
        if result and line and not line.startswith((" ", "\t", "#", "@")):
            break
        result.append(line)
    return "\n".join(result)


# ── 代码执行评测 ──────────────────────────────────────────

def check_correctness(problem: dict, completion: str, timeout: int = 5) -> bool:
    """执行生成的代码并检查测试用例是否通过"""
    # [FIX #9] 确保 completion 和 test 之间有空行
    test_code = (
        problem["prompt"]
        + completion.rstrip("\n") + "\n\n"
        + problem["test"] + "\n"
        + f"check({problem['entry_point']})"
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception as e:
        # [FIX #17] 区分超时和其他错误
        print(f"    [WARN] check_correctness infrastructure error: {type(e).__name__}: {e}")
        return False
    finally:
        os.unlink(tmp_path)


# ── Pass@k 计算 ──────────────────────────────────────────

def pass_at_k(n: int, c: int, k: int) -> float:
    """计算 pass@k (Chen et al. 2021 unbiased estimator)

    n=总样本数, c=通过数, k=k
    pass@k = 1 - C(n-c, k) / C(n, k)
    """
    # [FIX #4] 修正公式
    if n - c < k:
        return 1.0
    return 1.0 - np.prod([(n - c - i) / (n - i) for i in range(k)])


# ── 自适应采样 ──────────────────────────────────────────────

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
) -> tuple[list[str], list[bool], float, float, int]:
    """自适应采样：达到置信度阈值即早停

    返回 (completions, passed_list, mean_ms_per_token, total_time_ms, total_tokens)
    """
    completions = []
    passed_list = []
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
        passed = check_correctness(problem, comp)
        passed_list.append(passed)
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
    return completions, passed_list, mean_mspt, total_time_ms, total_tokens, agg_energy_mj


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
) -> BenchmarkResult:
    """运行一个完整的 HumanEval 评测配置"""

    print(f"\n{'='*60}")
    print(f"Config: {config_id}")
    print(f"  Model: {model_path}")
    print(f"  Precision: {precision}, Decoding: {decoding}, Sampling: {sampling}")
    print(f"{'='*60}")

    # [FIX #12] 重置显存统计
    torch.cuda.reset_peak_memory_stats()

    # 加载模型
    print("Loading model...")
    model, tokenizer = load_model(model_path, precision, device)

    # [FIX #2] 草稿模型始终用 FP16，不跟随主模型精度
    assistant_model = None
    assistant_tokenizer = None
    if decoding == "speculative" and draft_model_path:
        print(f"Loading draft model (FP16): {draft_model_path}")
        assistant_model, assistant_tokenizer = load_model(draft_model_path, "fp16", device)

    # 加载数据
    problems = load_humaneval()
    print(f"Loaded {len(problems)} HumanEval problems")

    results = []
    all_times = []

    for i, problem in enumerate(problems):
        task_id = problem["task_id"]
        prompt = problem["prompt"]

        if sampling == "greedy":
            comp, mspt, total_ms, n_tok, e_mj = generate_completion(
                model, tokenizer, prompt,
                temperature=0.0,
                assistant_model=assistant_model,
                assistant_tokenizer=assistant_tokenizer,
                device=device,
            )
            passed = check_correctness(problem, comp)

            gr = GenerationResult(
                task_id=task_id, prompt=prompt, completion=comp,
                passed=passed, ms_per_token=mspt, total_time_ms=total_ms,
                tokens_generated=n_tok,
                tokens_per_sec=n_tok / (total_ms / 1000) if total_ms > 0 else 0,
                energy_mj=e_mj,
            )
            results.append(gr)
            all_times.append(total_ms)

        elif sampling == "adaptive":
            comps, passed_list, mean_mspt, total_ms, total_tokens, e_mj = adaptive_sampling(
                model, tokenizer, prompt, problem,
                n=10, temperature=0.8,
                assistant_model=assistant_model,
                assistant_tokenizer=assistant_tokenizer,
                device=device,
            )
            best_idx = next((j for j, p in enumerate(passed_list) if p), 0)

            gr = GenerationResult(
                task_id=task_id, prompt=prompt, completion=comps[best_idx],
                passed=any(passed_list), ms_per_token=mean_mspt, total_time_ms=total_ms,
                tokens_generated=total_tokens,
                tokens_per_sec=total_tokens / (total_ms / 1000) if total_ms > 0 else 0,
                energy_mj=e_mj,
            )
            results.append(gr)
            all_times.append(total_ms)

        status = "PASS" if results[-1].passed else "FAIL"
        print(f"  [{i+1:3d}/{len(problems)}] {task_id}: {status} "
              f"({results[-1].total_time_ms:.0f}ms, {results[-1].tokens_generated} tok)")

    # 汇总
    peak_mem = get_peak_memory_mb(device)
    times = np.array(all_times)
    pass_count = sum(1 for r in results if r.passed)
    n_problems = len(results)

    # [FIX #5] 计算 pass@10（仅对 adaptive 有意义）
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
    )

    # 保存结果
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
    print(f"  Saved to {out_path}")

    # [FIX #10] 正确清理显存，防止 OOM
    del model
    if assistant_model is not None:
        del assistant_model
    if assistant_tokenizer is not None:
        del assistant_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return benchmark
