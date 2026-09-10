#!/bin/bash
# Watcher: 当 composition run{N} 的 C12.json 出现，就把 pareto run{N+1} 切到对应空闲 GPU。
# run1(GPU3, 现役) 继续跑；等 composition run1/run2 结束后启动 pareto run2(GPU0) / run3(GPU1)。
set -u
cd ~/efficient-codegen-exp
POLL=60
LOG=~/efficient-codegen-exp/launch-pareto-parallel.log

launch() {
  local dev=$1 run=$2 logf=$3
  if [ -f ~/efficient-codegen-exp/.pareto_${run}_launched ]; then return 0; fi
  nohup bash -c "bash run.sh scripts/run_pareto.py --device $dev --run $run && echo PARETO_RUN${run}_DONE" \
    > $logf 2>&1 &
  local pid=$!
  touch ~/efficient-codegen-exp/.pareto_${run}_launched
  echo "[$(date '+%F %T')] launched pareto run$run on $dev (PID $pid) -> $logf"
}

{
  echo "[$(date '+%F %T')] watcher started; poll=${POLL}s"
  while true; do
    # composition run1 完 -> pareto run2 到 GPU0
    [ -f results/composition/run1/C12.json ] && launch cuda:0 2 pareto_run2.log
    # composition run2 完 -> pareto run3 到 GPU1
    [ -f results/composition/run2/C12.json ] && launch cuda:1 3 pareto_run3.log
    # 两个都启动，退出
    if [ -f .pareto_2_launched ] && [ -f .pareto_3_launched ]; then
      echo "[$(date '+%F %T')] both pareto run2/run3 launched; watcher exiting."
      break
    fi
    latest_c=$(ls -t results/composition/run*/C*.json 2>/dev/null | head -1)
    echo "[$(date '+%F %T')] waiting; latest composition file: $(basename ${latest_c:-none})"
    sleep $POLL
  done
} >> $LOG 2>&1
