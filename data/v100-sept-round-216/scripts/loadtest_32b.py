#!/usr/bin/env python3
"""E3 gate: actual int4 load test of 32B on a V100. Writes verdict JSON + prints GATE lines.

headroom = device_total - torch max_memory_reserved after a short generate.
GATE GO   if headroom >= 1500 MiB  -> run P17 (32B int4 standard)
SPEC GO   if headroom >= 3000 MiB  -> also run P19 (32B int4 + 0.5B fp16 draft)
"""
import json, sys, time
sys.path.insert(0, "/root/efficient-codegen-exp/scripts")
import torch
from eval_humaneval import load_model

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
OUT = "/root/efficient-codegen-exp/results/heplus-x32/loadtest_32b.json"
res = {"device": DEV, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
try:
    t0 = time.time()
    model, tok = load_model("/root/efficient-codegen-exp/models/Qwen2.5-Coder-32B-Instruct", "int4", DEV)
    res["load_s"] = round(time.time() - t0, 1)
    inp = tok("def quicksort(arr):", return_tensors="pt").to(DEV)
    out = model.generate(**inp, max_new_tokens=16, do_sample=False)
    res["gen_tokens"] = int(out.shape[1])
    reserved_mb = round(torch.cuda.max_memory_reserved(DEV) / 1048576, 1)
    total_mb = round(torch.cuda.get_device_properties(DEV).total_memory / 1048576, 1)
    headroom_mb = round(total_mb - reserved_mb, 1)
    res.update(reserved_mb=reserved_mb, total_mb=total_mb, headroom_mb=headroom_mb)
    del model
    torch.cuda.empty_cache()
    res["gate"] = "GO" if headroom_mb >= 1500 else "SKIP"
    res["spec_gate"] = "GO" if headroom_mb >= 3000 else "SKIP"
except Exception as e:
    res["error"] = repr(e)[:500]
    res["gate"] = "SKIP"
    res["spec_gate"] = "SKIP"
import pathlib
pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(OUT).write_text(json.dumps(res, indent=2))
print("GATE:", res["gate"], "SPEC:", res["spec_gate"], json.dumps(res), flush=True)
