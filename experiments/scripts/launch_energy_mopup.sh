#!/bin/bash
cd ~/efficient-codegen-exp
nohup bash run.sh scripts/run_energy_round.py --device cuda:0 --only C08 > energy_c08_retry.log 2>&1 < /dev/null &
nohup bash run.sh scripts/run_energy_round.py --device cuda:2 --only C10 > energy_c10_retry.log 2>&1 < /dev/null &
nohup bash run.sh scripts/run_energy_round.py --device cuda:3 --only C12 > energy_c12_retry.log 2>&1 < /dev/null &
echo "launched 3 jobs at $(date)"
