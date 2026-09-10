#!/bin/bash
# 批量运行所有实验：3轮 Composition + 3轮 Pareto
# 用法: bash run_all.sh [--skip-composition] [--skip-pareto] [--device cuda:0]

set -e
cd ~/efficient-codegen-exp

DEVICE="cuda:0"
RUN_COMP=1
RUN_PARETO=1

for arg in "$@"; do
    case $arg in
        --skip-composition) RUN_COMP=0 ;;
        --skip-pareto) RUN_PARETO=0 ;;
        --device=*) DEVICE="${arg#*=}" ;;
    esac
done

run() {
    echo ""
    echo "========================================"
    echo " $1"
    echo " Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    bash run.sh "${@:2}"
    echo " Finished: $(date '+%Y-%m-%d %H:%M:%S')"
}

if [ $RUN_COMP -eq 1 ]; then
    for r in 1 2 3; do
        run "Composition Run $r/3" scripts/run_composition.py --device $DEVICE --run $r
    done
fi

if [ $RUN_PARETO -eq 1 ]; then
    for r in 1 2 3; do
        run "Pareto Run $r/3" scripts/run_pareto.py --device $DEVICE --run $r
    done
fi

echo ""
echo "========================================"
echo " ALL EXPERIMENTS COMPLETE"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
