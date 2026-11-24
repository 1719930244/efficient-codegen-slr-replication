#!/usr/bin/env python3
"""BigCodeBench 官方 instruct 协议复现: chat template 包裹 instruct_prompt, 模型续写函数体。

修复正文声明的口径偏离(completion prompt 配 Instruct 模型)。取文件前 300 题子集。
配置 B01-B04 = Qwen2.5-Coder-7B-Instruct fp16/int8/int4 standard greedy + fp16 speculative greedy。
判题复用 eval_bigcodebench.check_correctness(complete_prompt + completion + test)。
用法: python run_bcb_instruct_gen.py --device cuda:0 --ids B01 B02 B03 B04
输出: results/bcb_instruct/<ID>.jsonl(逐题 pass/latency) + <ID>_meta.json(聚合)
"""
import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import torch  # noqa: E402
from eval_humaneval import load_model, generate_completion, truncate_completion  # noqa: E402
from eval_bigcodebench import check_correctness_instruct  # noqa: E402

MODEL_DIR = Path.home() / "efficient-codegen-exp" / "models"
DRAFT = "Qwen2.5-Coder-0.5B-Instruct"
DATA = Path.home() / "efficient-codegen-exp" / "data" / "bigcodebench_instruct.jsonl"
NSUB = 300

CONFIGS = {
    "B01": ("Qwen2.5-Coder-7B-Instruct", "fp16", "standard"),
    "B02": ("Qwen2.5-Coder-7B-Instruct", "int8", "standard"),
    "B03": ("Qwen2.5-Coder-7B-Instruct", "int4", "standard"),
    "B04": ("Qwen2.5-Coder-7B-Instruct", "fp16", "speculative"),
}


def load_tasks():
    return [json.loads(l) for l in open(DATA, encoding="utf-8")][:NSUB]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ids", nargs="+", required=True)
    args = ap.parse_args()
    out_dir = Path.home() / "efficient-codegen-exp" / "results" / "bcb_instruct"
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks()

    groups = OrderedDict()
    for cid in args.ids:
        m, p, d = CONFIGS[cid]
        groups.setdefault((m, p), []).append(cid)

    for (mdir, prec), cids in groups.items():
        print(f"[LOAD] {mdir} {prec} on {args.device}", flush=True)
        model, tokenizer = load_model(str(MODEL_DIR / mdir), prec, args.device)
        for cid in cids:
            _, _, decoding = CONFIGS[cid]
            out_f = out_dir / f"{cid}.jsonl"
            done = set()
            if out_f.exists():
                for line in open(out_f, encoding="utf-8"):
                    done.add(json.loads(line)["task_id"])
                if len(done) >= len(tasks):
                    print(f"[SKIP] {cid} complete", flush=True)
                    continue
            draft = draft_tok = None
            if decoding == "speculative":
                draft, draft_tok = load_model(str(MODEL_DIR / DRAFT), "fp16", args.device)
            print(f"[RUN] {cid} remaining={len(tasks)-len(done)}", flush=True)
            npass = 0
            lats = []
            with open(out_f, "a", encoding="utf-8") as fo:
                for i, prob in enumerate(tasks):
                    tid = prob["task_id"]
                    if tid in done:
                        continue
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prob["instruct_prompt"]}],
                        tokenize=False, add_generation_prompt=True)
                    comp, _mspt, t_ms, ntok, _e = generate_completion(
                        model, tokenizer, prompt, temperature=0.0,
                        assistant_model=draft, assistant_tokenizer=draft_tok,
                        device=args.device, truncate=False)
                    passed, reason = check_correctness_instruct(prob, comp)
                    npass += 1 if passed else 0
                    lats.append(t_ms)
                    fo.write(json.dumps({"task_id": tid, "pass": bool(passed),
                                         "fail_reason": reason or "",
                                         "completion": comp,
                                         "latency_ms": t_ms, "tokens": ntok},
                                        ensure_ascii=False) + "\n")
                    fo.flush()
                    if (i + 1) % 25 == 0:
                        print(f"  [{cid} {i+1}/{len(tasks)}] {tid} pass={passed} {t_ms:.0f}ms", flush=True)
            n_done = len(lats)
            meta = {"config_id": cid, "model": mdir, "precision": prec, "decoding": decoding,
                    "benchmark": "bigcodebench-instruct-300",
                    "pass_at_1": round(100.0 * npass / n_done, 2) if n_done else None,
                    "p50_latency_ms": int(sorted(lats)[len(lats)//2]) if lats else None,
                    "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            (out_dir / f"{cid}_meta.json").write_text(json.dumps(meta, indent=2))
            print(f"[DONE] {cid} pass@1={meta['pass_at_1']} p50={meta['p50_latency_ms']}", flush=True)
            if draft is not None:
                del draft, draft_tok
                torch.cuda.empty_cache()
        del model, tokenizer
        torch.cuda.empty_cache()
    print("=== BCB-INSTRUCT GEN ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()
