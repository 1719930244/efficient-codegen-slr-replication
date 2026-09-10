#!/usr/bin/env python3
"""投机解码 draft 接受率测量（审稿人2: "logging acceptance rates 很便宜, 却是叙事而非数据"）。

复刻 HF assisted-decoding 的贪心接受规则: draft(0.5B FP16) 连出 K 个 token,
target 一次前向验证, 接受最长匹配前缀, 外加 1 个 bonus token。
统计: 接受率 = accepted/proposed; τ = 每轮平均生效 token 数(含 bonus)。

用法:
    python measure_acceptance.py --device cuda:0 --config T7FP16 [--n-problems 60] [--k 5]
输出:
    results/acceptance/<CONFIG>.json
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import torch  # noqa: E402
from eval_humaneval import load_model, load_humaneval  # noqa: E402

MODEL_DIR = Path.home() / "efficient-codegen-exp" / "models"
CONFIGS = {
    # key -> (target_dir, precision, draft_dir)
    "T7FP16":  ("Qwen2.5-Coder-7B-Instruct",  "fp16", "Qwen2.5-Coder-0.5B-Instruct"),
    "T7INT8":  ("Qwen2.5-Coder-7B-Instruct",  "int8", "Qwen2.5-Coder-0.5B-Instruct"),
    "T7INT4":  ("Qwen2.5-Coder-7B-Instruct",  "int4", "Qwen2.5-Coder-0.5B-Instruct"),
    "T32INT4": ("Qwen2.5-Coder-32B-Instruct", "int4", "Qwen2.5-Coder-0.5B-Instruct"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--n-problems", type=int, default=60,
                    help="固定取前 N 题(确定性切片), 默认 60")
    ap.add_argument("--k", type=int, default=5, help="每轮 draft 提议 token 数")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir or (Path.home() / "efficient-codegen-exp" / "results" / "acceptance"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_f = out_dir / f"{args.config}.json"
    if out_f.exists():
        print(f"[SKIP] {out_f} exists", flush=True)
        return

    tdir, prec, ddir = CONFIGS[args.config]
    print(f"[LOAD] target {tdir} {prec} + draft {ddir} on {args.device}", flush=True)
    target, tok = load_model(str(MODEL_DIR / tdir), prec, args.device)
    draft, dtok = load_model(str(MODEL_DIR / ddir), "fp16", args.device)
    dev = next(target.parameters()).device
    ddev = next(draft.parameters()).device
    eos = tok.eos_token_id

    problems = load_humaneval()[: args.n_problems]
    tot_prop = tot_acc = tot_rounds = 0
    taus = []
    t_start = time.time()

    for pi, prob in enumerate(problems):
        seq = tok(prob["prompt"], return_tensors="pt").input_ids.to(dev)
        n0 = seq.shape[1]
        p_prop = p_acc = p_rounds = 0
        while seq.shape[1] - n0 < 512:
            budget = min(args.k, 512 - (seq.shape[1] - n0))
            if budget <= 0:
                break
            dseq = seq.to(ddev)
            with torch.no_grad():
                dout = draft.generate(dseq, max_new_tokens=budget, do_sample=False,
                                      pad_token_id=dtok.pad_token_id)
            dtoks = dout[0][dseq.shape[1]:]
            if dtoks.numel() == 0:
                break
            dtoks = dtoks.to(dev)
            cand = torch.cat([seq[0], dtoks])
            with torch.no_grad():
                logits = target(input_ids=cand.unsqueeze(0)).logits[0]
            ttoks = logits[seq.shape[1] - 1: seq.shape[1] - 1 + dtoks.numel()].argmax(-1)
            match = (ttoks == dtoks)
            nacc = int(match.cumprod(0).sum().item())
            p_prop += int(dtoks.numel())
            p_acc += nacc
            p_rounds += 1
            newtoks = torch.cat([dtoks[:nacc], ttoks[nacc:nacc + 1]])
            seq = torch.cat([seq, newtoks.unsqueeze(0)], dim=1)
            if (newtoks == eos).any():
                break
        tot_prop += p_prop
        tot_acc += p_acc
        tot_rounds += p_rounds
        tau = (p_acc + p_rounds) / p_rounds if p_rounds else 0.0
        taus.append(tau)
        print(f"  [{pi+1}/{len(problems)}] {prob['task_id']} rounds={p_rounds} "
              f"acc={p_acc/max(p_prop,1):.3f} tau={tau:.2f} gen={seq.shape[1]-n0}", flush=True)

    n = len(taus)
    mean_tau = sum(taus) / n if n else 0.0
    sd_tau = math.sqrt(sum((x - mean_tau) ** 2 for x in taus) / (n - 1)) if n > 1 else 0.0
    res = {
        "config": args.config, "target": tdir, "precision": prec, "draft": ddir,
        "k": args.k, "n_problems": n,
        "acceptance_rate": tot_acc / max(tot_prop, 1),
        "tau_mean_tokens_per_round": mean_tau,
        "tau_sd": sd_tau,
        "total_proposed": tot_prop, "total_accepted": tot_acc, "total_rounds": tot_rounds,
        "wall_s": time.time() - t_start,
        "note": "instrumented draft-verify loop replicating HF assisted-decoding greedy acceptance rule; K fixed at args.k",
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_f.write_text(json.dumps(res, indent=2))
    print("[RESULT] " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
