#!/usr/bin/env bash
set -u
cd /root/efficient-codegen-exp/scripts
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/root/lcb-venv/bin/python
R=../results/lcb
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
if [ "$used" -gt 2000 ]; then echo "ABORT: GPUs busy (${used} MiB used)"; exit 1; fi
echo "[pipeline] start $(date -u)"
$PY run_lcb_gen.py --device cuda:0 --ids L02       > $R/gen_L02.log 2>&1 &
P1=$!
$PY run_lcb_gen.py --device cuda:1 --ids L01 L04 L03 > $R/gen_L01L04L03.log 2>&1 &
P2=$!
echo "[pipeline] gen pids: L02=$P1 (cuda:0)  L01L04L03=$P2 (cuda:1)"
wait $P1; R1=$?; wait $P2; R2=$?
echo "[pipeline] gen exited rc L02=$R1 L01L04L03=$R2 $(date -u)"
# resume-retry pass (skips complete configs instantly via jsonl/meta gate)
$PY run_lcb_gen.py --device cuda:0 --ids L02          >> $R/gen_L02.log 2>&1
$PY run_lcb_gen.py --device cuda:1 --ids L01 L04 L03  >> $R/gen_L01L04L03.log 2>&1
echo "[pipeline] retry done $(date -u)"
$PY judge_lcb.py --ids L01 L02 L03 L04 --parallel 16 --timeout 6 > $R/judge.log 2>&1
echo "[pipeline] judge rc=$? $(date -u)"
cat $R/lcb_summary.json
