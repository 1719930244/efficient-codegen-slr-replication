#!/bin/bash
# run2 接力链：MBPP run1 完成后 GPU0 接 MBPP run2；BCB run1 完成后 GPU1 接 BCB run2
set -u
cd ~/efficient-codegen-exp
LOG=~/exp_launch.log
PY=/home/szw/exp/bin/python
echo "[$(date)] run2接力watcher启动" >> $LOG

# 1) MBPP run1 -> run2
while [ ! -f results/mbpp/run1/C04.json ]; do sleep 300; done
sleep 30
echo "[$(date)] MBPP run1完成, 启动 MBPP run2(C01-C04, GPU0)" >> $LOG
nohup env PYTHONUNBUFFERED=1 $PY scripts/eval_mbpp.py --configs C01 C02 C03 C04 --device cuda:0 --run 2 \
  > ~/exp_mbpp_run2.log 2>&1 < /dev/null &

# 2) BCB run1 -> run2
while [ ! -f results/bigcodebench/run1/C04.json ]; do sleep 600; done
sleep 30
echo "[$(date)] BCB run1完成, 启动 BCB run2(C01-C04, GPU1)" >> $LOG
nohup env PYTHONUNBUFFERED=1 $PY scripts/eval_bigcodebench.py \
  --data /home/szw/efficient-codegen-exp/data/bigcodebench_full.jsonl \
  --configs C01 C02 C03 C04 --device cuda:1 --run 2 \
  > ~/exp_bcb_run2.log 2>&1 < /dev/null &

echo "[$(date)] run2接力链全部启动完毕, watcher退出" >> $LOG
