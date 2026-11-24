#!/bin/bash
# Energy Round Watcher v2
# - Skip C08/C11 in composition (extrapolated later).
# - Parallelize pareto across GPU2 (small P01-P09) + GPU3 (large P10-P15).
# - Mop-up on GPU0/GPU1 once pareto run2/run3 complete.
set -u
cd ~/efficient-codegen-exp
POLL=120
LOG=~/efficient-codegen-exp/launch-energy-round.log

COMP_SKIP=(C08 C11)
COMP_KEEP=(C01 C02 C03 C04 C05 C06 C07 C09 C10 C12)
PARETO_SMALL=(P01 P02 P03 P04 P05 P06 P07 P08 P09)
PARETO_LARGE=(P10 P11 P12 P13 P14 P15)
PARETO_ALL=(P01 P02 P03 P04 P05 P06 P07 P08 P09 P10 P11 P12 P13 P14 P15)

launch_comp_gpu3() {
  [ -f .energy_comp_launched ] && return 0
  nohup bash -c "bash run.sh scripts/run_energy_round.py --device cuda:3 --only ${COMP_KEEP[*]} && echo ENERGY_COMP_DONE" \
    > energy_comp.log 2>&1 &
  touch .energy_comp_launched
  echo "[$(date '+%F %T')] GPU3: energy composition ${COMP_KEEP[*]} launched (PID $!)"
}

launch_pareto_small_gpu2() {
  [ -f .energy_pareto_small_launched ] && return 0
  nohup bash -c "bash run.sh scripts/run_energy_round.py --device cuda:2 --only ${PARETO_SMALL[*]} && echo ENERGY_PARETO_SMALL_DONE" \
    > energy_pareto_small.log 2>&1 &
  touch .energy_pareto_small_launched
  echo "[$(date '+%F %T')] GPU2: energy pareto ${PARETO_SMALL[*]} launched (PID $!)"
}

launch_pareto_large_gpu3() {
  [ -f .energy_pareto_large_launched ] && return 0
  nohup bash -c "bash run.sh scripts/run_energy_round.py --device cuda:3 --only ${PARETO_LARGE[*]} && echo ENERGY_PARETO_LARGE_DONE" \
    > energy_pareto_large.log 2>&1 &
  touch .energy_pareto_large_launched
  echo "[$(date '+%F %T')] GPU3: energy pareto ${PARETO_LARGE[*]} launched (PID $!)"
}

# Mop-up: find configs still missing AND not already in-flight on another GPU.
# In-flight = launched via any _launched flag.
mop_up_gpu() {
  local gpu="$1" tag="$2"
  local flag=".energy_mopup_${tag}_launched"
  [ -f "$flag" ] && return 0
  local missing=()
  # Composition targets (already in-flight on GPU3 via .energy_comp_launched; mop-up only if that flag absent)
  if [ ! -f .energy_comp_launched ]; then
    for cid in "${COMP_KEEP[@]}"; do
      [ ! -f "results/energy_round/composition/${cid}.json" ] && missing+=("$cid")
    done
  fi
  # Pareto small: in-flight on GPU2 if small flag set; pareto large: in-flight on GPU3 if large flag set
  if [ ! -f .energy_pareto_small_launched ]; then
    for cid in "${PARETO_SMALL[@]}"; do
      [ ! -f "results/energy_round/pareto/${cid}.json" ] && missing+=("$cid")
    done
  fi
  if [ ! -f .energy_pareto_large_launched ]; then
    for cid in "${PARETO_LARGE[@]}"; do
      [ ! -f "results/energy_round/pareto/${cid}.json" ] && missing+=("$cid")
    done
  fi
  if [ ${#missing[@]} -eq 0 ]; then
    echo "[$(date '+%F %T')] $tag ($gpu): no missing configs; skip mop-up."
    touch "$flag"
    return 0
  fi
  nohup bash -c "bash run.sh scripts/run_energy_round.py --device $gpu --only ${missing[*]} && echo ENERGY_MOPUP_${tag}_DONE" \
    > "energy_mopup_${tag}.log" 2>&1 &
  touch "$flag"
  echo "[$(date '+%F %T')] $tag ($gpu): mop-up launched for ${missing[*]} (PID $!)"
}

all_done() {
  grep -q 'ENERGY_COMP_DONE' energy_comp.log 2>/dev/null || return 1
  grep -q 'ENERGY_PARETO_SMALL_DONE' energy_pareto_small.log 2>/dev/null || return 1
  grep -q 'ENERGY_PARETO_LARGE_DONE' energy_pareto_large.log 2>/dev/null || return 1
  return 0
}

{
  echo "[$(date '+%F %T')] energy watcher v2 started; poll=${POLL}s"
  echo "[$(date '+%F %T')] plan: GPU3=comp ${COMP_KEEP[*]} then large ${PARETO_LARGE[*]}; GPU2=small ${PARETO_SMALL[*]}; skip ${COMP_SKIP[*]}."
  while true; do
    [ -f results/pareto/run1/P15.json ] && launch_comp_gpu3
    [ -f results/composition/run3/C12.json ] && launch_pareto_small_gpu2
    if [ -f .energy_comp_launched ] && grep -q 'ENERGY_COMP_DONE' energy_comp.log 2>/dev/null; then
      launch_pareto_large_gpu3
    fi
    [ -f results/pareto/run2/P15.json ] && mop_up_gpu cuda:0 gpu0
    [ -f results/pareto/run3/P15.json ] && mop_up_gpu cuda:1 gpu1

    all_launched=true
    for f in .energy_comp_launched .energy_pareto_small_launched .energy_pareto_large_launched; do
      [ ! -f "$f" ] && all_launched=false
    done
    if $all_launched && all_done; then
      echo "[$(date '+%F %T')] all energy rounds launched AND finished; watcher exiting."
      break
    fi
    sleep $POLL
  done
} >> $LOG 2>&1
