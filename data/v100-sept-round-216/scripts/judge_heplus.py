#!/usr/bin/env python3
"""EvalPlus (HumanEval+) 判题: results/heplus/<ID>.jsonl -> 每配置 base/plus 通过率。

用法(在装有 evalplus-venv 的机器上):
    python judge_heplus.py                 # 判所有完整 jsonl
    python judge_heplus.py --ids C08 P10   # 只判指定配置
输出: results/heplus_summary.json (合并累积)
说明: greedy 配置 solution_list 长度=1, pass@1 即正确率;
      adaptive 配置 solution_list<=10, 报告 any-pass 与无偏 pass@1 两种口径。
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HEPLUS = Path.home() / "efficient-codegen-exp" / "results" / "heplus"
VENV = Path.home() / "evalplus-venv" / "bin" / "python"


def expand(jsonl_path):
    rows = []
    for line in open(jsonl_path, encoding="utf-8"):
        r = json.loads(line)
        sols = r.get("solution_list")
        if not sols:
            sols = [r.get("completion", r.get("solution", ""))]
        for sol in sols:
            rows.append({"task_id": r["task_id"], "completion": sol})
    return rows


def statuses_all_pass(statuses):
    return all(s == "pass" for s in statuses)


def any_pass(statuses):
    return any(s == "pass" for s in statuses)


def pass_at_k_unbiased(n, c, k=1):
    """unbiased estimator (Chen et al. 2021)"""
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def judge_one(cid):
    src = HEPLUS / f"{cid}.jsonl"
    n_tasks = sum(1 for _ in open(src, encoding="utf-8"))
    rows = expand(src)
    tmp = HEPLUS / f"{cid}.evalplus-samples.jsonl"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    outdir = HEPLUS / "evalout"
    outdir.mkdir(exist_ok=True)
    res_f = tmp.with_name(tmp.name + "_eval_results.json")
    if res_f.exists():
        res_f.unlink()
    env = dict(os.environ, HF_ENDPOINT="https://hf-mirror.com")
    cmd = [str(VENV), "-m", "evalplus.evaluate", "--dataset", "humaneval",
           "--samples", str(tmp),
           "--i-just-wanna-run", "--test-details", "--parallel", "16"]
    print(f"[JUDGE] {cid}: {n_tasks} tasks, {len(rows)} solutions", flush=True)
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, input="y\ny\ny\n")
    if p.returncode != 0:
        print(f"[JUDGE-ERR] {cid} rc={p.returncode}\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}", flush=True)
        return None
    if not res_f.exists():
        # evalplus 可能把结果写到 samples 同目录的另一命名
        cands = list(HEPLUS.glob(f"{cid}*eval_results.json")) + list(outdir.glob("*eval_results.json"))
        if not cands:
            print(f"[JUDGE-ERR] {cid}: no result json found", flush=True)
            return None
        res_f = sorted(cands)[-1]
    res = json.loads(res_f.read_text())
    ev = res.get("eval", {})
    base_p1 = plus_p1 = base_any = plus_any = 0.0
    n = 0
    per_task = []
    for tid, d in ev.items():
        entries = d if isinstance(d, list) else [d]
        bs = [e.get("base_status") for e in entries]
        ps = [e.get("plus_status") for e in entries]
        if not bs:
            continue
        n += 1
        c_b = sum(1 for s in bs if s == "pass")
        c_p = sum(1 for s in ps if s == "pass")
        base_p1 += pass_at_k_unbiased(len(bs), c_b)
        plus_p1 += pass_at_k_unbiased(len(ps), c_p)
        base_any += 1.0 if any_pass(bs) else 0.0
        plus_any += 1.0 if any_pass(ps) else 0.0
        per_task.append({"task_id": tid, "base_pass": c_b, "base_n": len(bs),
                         "plus_pass": c_p, "plus_n": len(ps)})
    if n == 0:
        print(f"[JUDGE-ERR] {cid}: empty eval dict", flush=True)
        return None
    out = {
        "config_id": cid,
        "n_tasks": n,
        "n_solutions": len(rows),
        "base_pass_at_1": round(100.0 * base_p1 / n, 2),
        "plus_pass_at_1": round(100.0 * plus_p1 / n, 2),
        "base_any_pass": round(100.0 * base_any / n, 2),
        "plus_any_pass": round(100.0 * plus_any / n, 2),
    }
    print(f"[JUDGE-OK] {cid}: base@1={out['base_pass_at_1']} plus@1={out['plus_pass_at_1']} "
          f"base_any={out['base_any_pass']} plus_any={out['plus_any_pass']}", flush=True)
    detail_f = HEPLUS / f"{cid}.evalplus-detail.json"
    detail_f.write_text(json.dumps({"summary": out, "per_task": per_task}, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", default=None)
    args = ap.parse_args()
    summary_f = HEPLUS / "heplus_summary.json"
    summary = {}
    if summary_f.exists():
        summary = json.loads(summary_f.read_text())
    if args.ids:
        cids = args.ids
    else:
        cids = sorted(p.stem for p in HEPLUS.glob("*.jsonl")
                      if not p.name.startswith(".") and "samples" not in p.name)
    for cid in cids:
        src = HEPLUS / f"{cid}.jsonl"
        if not src.exists():
            print(f"[SKIP] {cid}: no jsonl", flush=True)
            continue
        meta_f = HEPLUS / f"{cid}_meta.json"
        if not meta_f.exists():
            print(f"[SKIP] {cid}: run not finished (no meta)", flush=True)
            continue
        if cid in summary:
            print(f"[SKIP] {cid}: already judged", flush=True)
            continue
        out = judge_one(cid)
        if out:
            summary[cid] = out
            summary_f.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print("=== JUDGE ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()
