#!/usr/bin/env python3
"""E3 step 2: verify 32B download completeness (shard count, sizes, no .incomplete, required files)."""
import json, sys
from pathlib import Path

d = Path("/root/lcb-models/Qwen2.5-Coder-32B-Instruct")
st = sorted(d.glob("model-*.safetensors"))
total = sum(f.stat().st_size for f in st)
inc = list(d.rglob("*.incomplete"))
need = ["config.json", "model.safetensors.index.json", "tokenizer.json",
        "tokenizer_config.json", "generation_config.json"]
miss = [n for n in need if not (d / n).exists()]
ok = len(st) == 14 and not inc and not miss
ts = None
if (d / "model.safetensors.index.json").exists():
    idx = json.load(open(d / "model.safetensors.index.json"))
    ts = idx.get("metadata", {}).get("total_size")
    if ts and abs(total - ts) / ts > 0.01:
        ok = False
print(f"shards={len(st)} bytes={total} index_total_size={ts} incomplete={len(inc)} missing={miss}")
print("VERIFY_OK" if ok else "VERIFY_FAIL")
sys.exit(0 if ok else 1)
