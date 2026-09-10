#!/usr/bin/env python3
"""LCB 判题: 官方 check_correctness @28fef95 (timeout=6/test, public+private),
官方 extract_code generic 分支 (最后一对 ``` 围栏; <2 围栏 -> "" 判负, 与官方一致).
Gate: <ID>_meta.json 存在且行数==n_tasks; lcb_summary.json 累积合并 (crash-safe)."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.set_int_max_str_digits(50000)
BACKEND = "official-import@28fef95"
try:
    from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness
except Exception as e:
    print(f"[WARN] official import failed ({e!r}); falling back to vendored copy", flush=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
    from lcb_check import check_correctness
    BACKEND = "vendored@28fef95"

def extract_code(text):
    # verbatim semantics of lcb_runner/utils/extraction_utils.py::extract_code (generic branch)
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if "```" in l]
    if len(idx) < 2:
        return ""
    return "\n".join(lines[idx[-2] + 1 : idx[-1]])

_PROBS = None; _TIMEOUT = 6
def _init(probs, timeout):
    global _PROBS, _TIMEOUT; _PROBS = probs; _TIMEOUT = timeout

def _judge(item):
    tid, gen = item
    try:
        res, _md = check_correctness({"input_output": _PROBS[tid]["input_output"]},
                                     gen, timeout=_TIMEOUT)
        fixed = []
        for e in res:
            if isinstance(e, np.ndarray): e = e.item()
            if isinstance(e, np.bool_): e = bool(e)
            fixed.append(e)
        passed = bool(fixed) and all(e is True for e in fixed)
        n_pt = sum(1 for e in fixed if e is True)
        err = None
    except Exception as e:
        fixed, passed, n_pt, err = [], False, 0, repr(e)
    return tid, {"passed": passed, "n_tests": _PROBS[tid]["n_tests"], "n_passed_tests": n_pt,
                 "extracted_empty": not gen.strip(), "judge_error": err}

def pct(a, b): return round(100.0 * a / b, 2) if b else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", default=["L01", "L02", "L03", "L04"])
    ap.add_argument("--dir", default="/root/efficient-codegen-exp/results/lcb")
    ap.add_argument("--problems", default="/root/efficient-codegen-exp/data/lcb_eval_samples.jsonl")
    ap.add_argument("--parallel", type=int, default=16)   # official num_process_evaluate default
    ap.add_argument("--timeout", type=int, default=6)     # official per-test default
    args = ap.parse_args()
    D = Path(args.dir)
    summary_f = D / "lcb_summary.json"
    summary = json.loads(summary_f.read_text()) if summary_f.exists() else {}
    probs = {r["task_id"]: r for r in map(json.loads, open(args.problems, encoding="utf-8"))}
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    # official check_correctness spawns a Process per test; mp.Pool workers are
    # daemonic and forbid children, so use ProcessPoolExecutor (non-daemon).
    pool = ProcessPoolExecutor(max_workers=args.parallel,
                               mp_context=mp.get_context("fork"),
                               initializer=_init, initargs=(probs, args.timeout))
    for cid in args.ids:
        if cid in summary: print(f"[SKIP] {cid} already in summary", flush=True); continue
        meta_f, src = D / f"{cid}_meta.json", D / f"{cid}.jsonl"
        if not meta_f.exists(): print(f"[SKIP] {cid}: meta missing (run not finished)", flush=True); continue
        meta = json.loads(meta_f.read_text())
        rows = [json.loads(l) for l in open(src, encoding="utf-8")]
        assert len(rows) == meta["n_tasks"], f"{cid}: {len(rows)} rows != meta {meta['n_tasks']}"
        gens = {r["task_id"]: extract_code(r["completion"]) for r in rows}
        assert set(gens) <= set(probs), f"{cid}: unknown task_ids"
        n_empty = sum(1 for g in gens.values() if not g.strip())
        t0 = time.time()
        results = dict(pool.map(_judge, sorted(gens.items())))
        n = len(results); npass = sum(1 for m in results.values() if m["passed"])
        per_task = []
        for tid in sorted(results):
            m = results[tid]
            per_task.append({"task_id": tid, "passed": m["passed"], "n_tests": m["n_tests"],
                "n_passed_tests": m["n_passed_tests"], "extracted_empty": m["extracted_empty"],
                "judge_error": m["judge_error"], "difficulty": probs[tid]["difficulty"],
                "platform": probs[tid]["platform"], "has_starter": probs[tid]["has_starter"]})
        by = lambda key: {v: pct(sum(1 for p in per_task if p[key] == v and p["passed"]),
                                 sum(1 for p in per_task if p[key] == v))
                          for v in sorted({p[key] for p in per_task})}
        lat = sorted(r["latency_ms"] for r in rows)
        toks = sum(r["tokens"] for r in rows)
        en = [r["energy_j"] for r in rows if r.get("energy_j") is not None]
        out = {"config_id": cid, "backend": BACKEND, "n_tasks": n,
            "pass_at_1": pct(npass, n), "by_difficulty": by("difficulty"), "by_platform": by("platform"),
            "n_with_starter": sum(1 for p in per_task if p["has_starter"]),
            "n_empty_extraction": n_empty, "n_at_cap": meta.get("n_at_cap"),
            "timeout_per_test_s": args.timeout, "test_split": "public+private",
            "mean_latency_ms": round(sum(lat)/len(lat), 1), "p50_latency_ms": lat[len(lat)//2],
            "p95_latency_ms": lat[int(0.95*(len(lat)-1))],
            "mean_tok_s": round(toks / (sum(lat)/1000.0), 2),
            "total_output_tokens": toks,
            "energy_total_j": round(sum(en), 1) if en else None,
            "energy_j_per_req": round(sum(en)/len(en), 2) if en else None,
            "peak_mem_mb": meta.get("peak_mem_mb"), "judge_wall_s": round(time.time()-t0, 1),
            "judged_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (D / f"{cid}.lcb-detail.json").write_text(
            json.dumps({"summary": out, "per_task": per_task}, indent=1, ensure_ascii=False))
        summary[cid] = out
        summary_f.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
        print(f"[JUDGED] {cid}: pass@1={out['pass_at_1']}% empty={n_empty} wall={out['judge_wall_s']}s", flush=True)
    pool.shutdown()

if __name__ == "__main__":
    main()
