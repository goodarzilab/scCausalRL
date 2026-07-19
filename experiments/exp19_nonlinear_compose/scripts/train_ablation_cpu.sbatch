#!/bin/bash
#SBATCH --job-name=abl_cpu
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=340G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ablcpu_%x_%j.out
#SBATCH --error=logs/ablcpu_%x_%j.err
set -euo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose
case "$CFG" in /*) ;; *) CFG="$REPO/$CFG" ;; esac
OUT="$REPO/results/grid/$TAG"; mkdir -p "$OUT" "$REPO/logs"; cd "$OUT"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"; export OMP_NUM_THREADS=16
echo "start $(date) on $(hostname) | CPU 60-epoch | cfg=$CFG tag=$TAG"
python -u "$REPO/scripts/run_phase1.py" --config "$CFG" --device cpu
echo "done $(date)"; ls -la checkpoints/ | tail -4
