#!/bin/bash
# 实验启动脚本：使用 micromamba exp 环境
cd ~/efficient-codegen-exp
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

~/.local/bin/micromamba run -n exp --root-prefix ~/micromamba python -u "$@"
