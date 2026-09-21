#!/usr/bin/env python3
"""E3 step 1: download Qwen2.5-Coder-32B-Instruct via hf-mirror to /root/lcb-models/."""
import os, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import snapshot_download

t0 = time.time()
p = snapshot_download(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    local_dir="/root/lcb-models/Qwen2.5-Coder-32B-Instruct",
    max_workers=4,
)
print("SNAPSHOT_DONE", p, "elapsed_s", round(time.time() - t0, 1), flush=True)
