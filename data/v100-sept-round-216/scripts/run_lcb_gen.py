#!/usr/bin/env python3
"""LiveCodeBench codegeneration 生成 — post-2024-11-01 contamination-free slice (N=288).
L01-L04 = Qwen2.5-Coder-7B-Instruct fp16/int8/int4 standard greedy + fp16 speculative greedy.
Prompt = 官方 LCB generic chat template, lcb_runner/prompts/code_generation.py @28fef95 (硬编码, 不 import).
无内联判题; judge_lcb.py 离线判. 用法: python run_lcb_gen.py --device cuda:0 --ids L02"""
import argparse, gc, json, sys, time
from collections import OrderedDict
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import torch, transformers
from eval_humaneval import load_model, generate_completion   # 带入 HF_ENDPOINT + FIX#21 spec patch
try:
    from eval_humaneval import get_peak_memory_mb
except ImportError:
    get_peak_memory_mb = lambda dev: round(torch.cuda.max_memory_reserved() / 1048576, 1)

MODEL_DIR = Path("/root/lcb-models")
TARGET = "Qwen2.5-Coder-7B-Instruct"
DRAFT  = "Qwen2.5-Coder-0.5B-Instruct"

SYSTEM_MESSAGE = ("You are an expert Python programmer. You will be given a question (problem "
                  "specification) and will generate a correct Python program that matches the "
                  "specification and passes all tests.")
FMT_STARTER = ("You will use the following starter code to write the solution to the problem "
               "and enclose your code within delimiters.")
FMT_STDIN = ("Read the inputs from stdin solve the problem and write the answer to stdout (do not "
             "directly test on the sample inputs). Enclose your code within delimiters as follows. "
             "Ensure that when the python program runs, it reads the inputs, runs the algorithm "
             "and writes output to STDOUT.")

def build_user(t):
    p = f"### Question:\n{t['question_content']}\n\n"
    if t["has_starter"]:
        p += f"### Format: {FMT_STARTER}\n```python\n{t['starter_code']}\n```\n\n"
    else:
        p += f"### Format: {FMT_STDIN}\n```python\n# YOUR CODE HERE\n```\n\n"
    p += "### Answer: (use the provided format with backticks)\n\n"
    return p

CONFIGS = OrderedDict([("L01", ("fp16", "standard")), ("L02", ("int8", "standard")),
                       ("L03", ("int4", "standard")), ("L04", ("fp16", "speculative"))])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--tasks", default="/root/efficient-codegen-exp/data/lcb_tasks.jsonl")
    ap.add_argument("--out-dir", default="/root/efficient-codegen-exp/results/lcb")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(l) for l in open(args.tasks, encoding="utf-8")]
    groups = OrderedDict()
    for cid in args.ids:
        groups.setdefault(CONFIGS[cid][0], []).append(cid)

    for prec, cids in groups.items():
        model, tokenizer = load_model(str(MODEL_DIR / TARGET), prec, args.device)
        for cid in cids:
            dec = CONFIGS[cid][1]
            out_f = out_dir / f"{cid}.jsonl"
            done = set()
            if out_f.exists():                      # robust resume (run_heplus_gen variant)
                for line in open(out_f, encoding="utf-8"):
                    try: done.add(json.loads(line)["task_id"])
                    except Exception: pass
                if len(done) >= len(tasks):
                    print(f"[SKIP] {cid} already complete", flush=True); continue
            draft = draft_tok = None
            if dec == "speculative":
                draft, draft_tok = load_model(str(MODEL_DIR / DRAFT), "fp16", args.device)  # FIX#2
            torch.cuda.reset_peak_memory_stats()                            # FIX#12
            n_at_cap = n_done = 0; energy_any = False; t0 = time.time()
            with open(out_f, "a", encoding="utf-8") as fo:
                for t in tasks:
                    tid = t["task_id"]
                    if tid in done: continue
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "system", "content": SYSTEM_MESSAGE},
                         {"role": "user",   "content": build_user(t)}],
                        tokenize=False, add_generation_prompt=True)
                    n_in = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                    comp, _mspt, t_ms, ntok, e_mj = generate_completion(
                        model, tokenizer, prompt, max_new_tokens=args.max_new_tokens,
                        temperature=0.0, assistant_model=draft, assistant_tokenizer=draft_tok,
                        device=args.device, truncate=False)     # whole-program, EOS-terminated
                    hit_cap = ntok >= args.max_new_tokens
                    n_at_cap += hit_cap; n_done += 1
                    if e_mj >= 0: energy_any = True
                    fo.write(json.dumps({"task_id": tid, "completion": comp, "tokens": ntok,
                        "latency_ms": t_ms, "input_tokens": n_in, "hit_cap": hit_cap,
                        "energy_j": round(e_mj/1000.0, 3) if e_mj >= 0 else None},
                        ensure_ascii=False) + "\n"); fo.flush()
                    if n_done % 25 == 0:
                        print(f"  [{cid} {n_done}] {tid} {t_ms:.0f}ms {ntok}tok", flush=True)
            meta = {"config_id": cid, "model": TARGET, "precision": prec, "decoding": dec,
                "sampling": "greedy", "benchmark": "livecodebench-codegen",
                "slice": "release_v6 contest_date>=2024-11-01 (N=288)", "n_tasks": len(tasks),
                "max_new_tokens": args.max_new_tokens,
                "prompt_mode": "official-chat-generic@28fef95",
                "device": args.device, "gpu_name": torch.cuda.get_device_name(),
                "torch": torch.__version__, "transformers": transformers.__version__,
                "n_at_cap": n_at_cap, "energy_measured": energy_any,
                "peak_mem_mb": get_peak_memory_mb(args.device),
                "wall_s": round(time.time() - t0, 1),
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            (out_dir / f"{cid}_meta.json").write_text(json.dumps(meta, indent=2))
            print(f"[DONE] {cid}: {n_done} rows, n_at_cap={n_at_cap}, wall={meta['wall_s']}s", flush=True)
            if draft is not None:
                del draft, draft_tok; draft = draft_tok = None; torch.cuda.empty_cache()
        del model, tokenizer; gc.collect(); torch.cuda.empty_cache()        # FIX#10

if __name__ == "__main__":
    main()
