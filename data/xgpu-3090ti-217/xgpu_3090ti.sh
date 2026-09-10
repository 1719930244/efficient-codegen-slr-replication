#!/bin/bash
cd /root/efficient-codegen-exp/scripts || exit 1
P=/home/v3090ti/anaconda3/envs/train/bin/python
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
L=/root/xgpu_3090ti.log
echo "[$(date -u)] 3090Ti cross-GPU chain start: P10 P11 P12 C04 C05 C06 + T7 acceptance" >> $L
for ID in P10 P11 P12 C04 C05 C06; do
  $P run_heplus_gen.py --device cuda:0 --ids $ID >> $L 2>&1
done
for T in T7FP16 T7INT8 T7INT4; do
  $P measure_acceptance.py --device cuda:0 --config $T >> $L 2>&1
done
echo "[$(date -u)] 3090TI ALL DONE" >> $L
