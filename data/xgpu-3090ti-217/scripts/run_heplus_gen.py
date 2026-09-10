#!/usr/bin/env python3
"""HumanEval 补生成: 保存每题 completions 到 JSONL, 供 EvalPlus 离线判题 (HumanEval/HumanEval+)。

审稿人2点名要求补 HumanEval+ (EvalPlus)。原 results 树未保存 completions(details 为空),
故按原协议(同 prompt/截断/采样参数)重新生成一遍并保存样本。原 results/ 树不动, 输出到 results/heplus/。

用法:
    python run_heplus_gen.py --device cuda:0 --ids C08 C09 P02
    # 支持断点续跑: <ID>.jsonl 已有的 task_id 跳过; 满 164 题整配置跳过
输出:
    results/heplus/<ID>.jsonl        {"task_id", "solution_list", "latency_ms", "tokens"} 每题一行
    results/heplus/<ID>_meta.json    配置元信息
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
from eval_humaneval import (  # noqa: E402
    load_model, generate_completion, adaptive_sampling, load_humaneval,
)

MODEL_DIR = Path.home() / "efficient-codegen-exp" / "models"
DRAFT = "Qwen2.5-Coder-0.5B-Instruct"
FAMILY2_DRAFT = "deepseek-coder-1.3b-instruct"

# ID -> (model_dir, precision, decoding, sampling)
# C01-C03 与 P10-P12 配置相同(7B std greedy), 只跑 P 侧一份, completions 通用。
CONFIGS = {
    "P01": ("Qwen2.5-Coder-0.5B-Instruct", "fp16", "standard", "greedy"),
    "P02": ("Qwen2.5-Coder-0.5B-Instruct", "int8", "standard", "greedy"),
    "P03": ("Qwen2.5-Coder-0.5B-Instruct", "int4", "standard", "greedy"),
    "P04": ("Qwen2.5-Coder-1.5B-Instruct", "fp16", "standard", "greedy"),
    "P05": ("Qwen2.5-Coder-1.5B-Instruct", "int8", "standard", "greedy"),
    "P06": ("Qwen2.5-Coder-1.5B-Instruct", "int4", "standard", "greedy"),
    "P07": ("Qwen2.5-Coder-3B-Instruct", "fp16", "standard", "greedy"),
    "P08": ("Qwen2.5-Coder-3B-Instruct", "int8", "standard", "greedy"),
    "P09": ("Qwen2.5-Coder-3B-Instruct", "int4", "standard", "greedy"),
    "P10": ("Qwen2.5-Coder-7B-Instruct", "fp16", "standard", "greedy"),
    "P11": ("Qwen2.5-Coder-7B-Instruct", "int8", "standard", "greedy"),
    "P12": ("Qwen2.5-Coder-7B-Instruct", "int4", "standard", "greedy"),
    "P13": ("Qwen2.5-Coder-14B-Instruct", "fp16", "standard", "greedy"),
    "P14": ("Qwen2.5-Coder-14B-Instruct", "int8", "standard", "greedy"),
    "P15": ("Qwen2.5-Coder-14B-Instruct", "int4", "standard", "greedy"),
    "P16": ("Qwen2.5-Coder-32B-Instruct", "int8", "standard", "greedy"),
    "P17": ("Qwen2.5-Coder-32B-Instruct", "int4", "standard", "greedy"),
    "P18": ("Qwen2.5-Coder-14B-Instruct", "fp16", "speculative", "greedy"),
    "P19": ("Qwen2.5-Coder-32B-Instruct", "int4", "speculative", "greedy"),
    "C04": ("Qwen2.5-Coder-7B-Instruct", "fp16", "speculative", "greedy"),
    "C05": ("Qwen2.5-Coder-7B-Instruct", "int8", "speculative", "greedy"),
    "C06": ("Qwen2.5-Coder-7B-Instruct", "int4", "speculative", "greedy"),
    "C07": ("Qwen2.5-Coder-7B-Instruct", "fp16", "standard", "adaptive"),
    "C08": ("Qwen2.5-Coder-7B-Instruct", "int8", "standard", "adaptive"),
    "C09": ("Qwen2.5-Coder-7B-Instruct", "int4", "standard", "adaptive"),
    "C10": ("Qwen2.5-Coder-7B-Instruct", "fp16", "speculative", "adaptive"),
    "C11": ("Qwen2.5-Coder-7B-Instruct", "int8", "speculative", "adaptive"),
    "C12": ("Qwen2.5-Coder-7B-Instruct", "int4", "speculative", "adaptive"),
    # 第二模型家族(审稿人2/3要求; 预注册外探索性补充)
    "F01": ("deepseek-coder-6.7b-instruct", "fp16", "standard", "greedy"),
    "F02": ("deepseek-coder-6.7b-instruct", "int8", "standard", "greedy"),
    "F03": ("deepseek-coder-6.7b-instruct", "int4", "standard", "greedy"),
    "F04": ("deepseek-coder-6.7b-instruct", "fp16", "speculative", "greedy"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out_dir or (Path.home() / "efficient-codegen-exp" / "results" / "heplus"))
    out_dir.mkdir(parents=True, exist_ok=True)

    problems = load_humaneval()
    n_prob = len(problems)

    # 按 (model, precision) 分组减少加载次数, 保持传入顺序
    groups = OrderedDict()
    for cid in args.ids:
        if cid not in CONFIGS:
            print(f"[ERR] unknown config {cid}", flush=True)
            continue
        m, p, _, _ = CONFIGS[cid]
        groups.setdefault((m, p), []).append(cid)

    for (mdir, prec), cids in groups.items():
        model_path = str(MODEL_DIR / mdir)
        print(f"[LOAD] {mdir} {prec} on {args.device}", flush=True)
        t0 = time.time()
        model, tokenizer = load_model(model_path, prec, args.device)
        print(f"[LOAD] done in {time.time()-t0:.0f}s", flush=True)

        for cid in cids:
            _, _, decoding, sampling = CONFIGS[cid]
            out_f = out_dir / f"{cid}.jsonl"
            done_ids = set()
            if out_f.exists():
                with open(out_f) as f:
                    for line in f:
                        try:
                            done_ids.add(json.loads(line)["task_id"])
                        except Exception:
                            pass
                if len(done_ids) >= n_prob:
                    print(f"[SKIP] {cid} complete ({len(done_ids)})", flush=True)
                    continue

            draft = draft_tok = None
            if decoding == "speculative":
                dname = FAMILY2_DRAFT if cid.startswith("F") else DRAFT
                print(f"[LOAD] draft {dname}", flush=True)
                draft, draft_tok = load_model(str(MODEL_DIR / dname), "fp16", args.device)

            print(f"[RUN] {cid} {mdir}/{prec}/{decoding}/{sampling} remaining={n_prob-len(done_ids)}", flush=True)
            with open(out_f, "a") as fo:
                for i, prob in enumerate(problems):
                    tid = prob["task_id"]
                    if tid in done_ids:
                        continue
                    if sampling == "greedy":
                        comp, _mspt, t_ms, ntok, _e = generate_completion(
                            model, tokenizer, prob["prompt"], temperature=0.0,
                            assistant_model=draft, assistant_tokenizer=draft_tok,
                            device=args.device)
                        comps, lat = [comp], t_ms
                    else:
                        comps, _pl, _fr, _mspt, t_ms, ntok, _e = adaptive_sampling(
                            model, tokenizer, prob["prompt"], prob,
                            n=10, temperature=0.8,
                            assistant_model=draft, assistant_tokenizer=draft_tok,
                            device=args.device)
                        lat = t_ms
                    rec = {"task_id": tid, "solution_list": comps,
                           "latency_ms": lat, "tokens": ntok}
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fo.flush()
                    print(f"  [{cid} {i+1}/{n_prob}] {tid} {lat:.0f}ms {ntok}tok n={len(comps)}", flush=True)

            meta = {"config_id": cid, "model": mdir, "precision": prec,
                    "decoding": decoding, "sampling": sampling, "device": args.device,
                    "n_problems": n_prob,
                    "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            (out_dir / f"{cid}_meta.json").write_text(json.dumps(meta, indent=2))
            print(f"[DONE] {cid}", flush=True)

            if draft is not None:
                del draft, draft_tok
                torch.cuda.empty_cache()

        del model, tokenizer
        torch.cuda.empty_cache()

    print("=== HEPLUS GEN ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()
