#!/bin/bash
# LCB L01 (FP16 std) energy/latency variance: 3 sequential full passes
# Protocol identical to the 2026-09-10 L01 run: cuda:1, 512-token cap, official chat prompt, 288 tasks.
set -x
cd /root/efficient-codegen-exp/scripts
V=/root/lcb-venv/bin/python
BASE=/root/efficient-codegen-exp/results/lcb-energy-variance
for i in 1 2 3; do
  # wait until GPU1 idle (max 1h)
  for w in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    [ "$used" -lt 1000 ] && break
    sleep 120
  done
  OUT=$BASE/run$i
  mkdir -p "$OUT"
  $V run_lcb_gen.py --device cuda:1 --ids L01 --out-dir "$OUT" > /root/efficient-codegen-exp/logs/ev_run$i.log 2>&1
  $V judge_lcb.py --ids L01 --dir "$OUT" > /root/efficient-codegen-exp/logs/ev_judge$i.log 2>&1
done
touch $BASE/DONE
