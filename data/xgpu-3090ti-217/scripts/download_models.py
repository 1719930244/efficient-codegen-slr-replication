"""下载 Qwen2.5-Coder-Instruct 系列模型（从 ModelScope）"""

import subprocess
import sys
from pathlib import Path

MODELS = [
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "Qwen/Qwen2.5-Coder-3B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen2.5-Coder-14B-Instruct",
]

MODEL_DIR = Path.home() / "efficient-codegen-exp" / "models"


def download_model(model_id: str):
    name = model_id.split("/")[1]
    target = MODEL_DIR / name
    if target.exists() and any(target.iterdir()):
        print(f"[SKIP] {name} already exists at {target}")
        return
    print(f"[DOWNLOAD] {model_id} -> {target}")
    from modelscope import snapshot_download
    snapshot_download(model_id, cache_dir=str(MODEL_DIR), local_dir=str(target))
    print(f"[DONE] {name}")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1:
        # 下载指定模型，如: python download_models.py 0.5B 7B
        sizes = sys.argv[1:]
        models = [m for m in MODELS if any(s in m for s in sizes)]
    else:
        models = MODELS

    for model_id in models:
        download_model(model_id)
    print("\n=== All downloads complete ===")


if __name__ == "__main__":
    main()
