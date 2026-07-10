#!/bin/bash
#SBATCH --job-name=exp20_of
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch
#SBATCH --mem=150G
#SBATCH --cpus-per-task=6
#SBATCH --time=16:00:00
#SBATCH --output=logs/of_%x_%j.out
#SBATCH --error=logs/of_%x_%j.err
set -euo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
CFG="${CFG:?set CFG}"; TAG="${TAG:?set TAG}"
case "$CFG" in /*) ;; *) CFG="$REPO/$CFG" ;; esac
OUT="$REPO/results/grid/$TAG"; mkdir -p "$OUT" "$REPO/logs"; cd "$OUT"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
echo "start $(date) on $(hostname) | cfg=$CFG tag=$TAG"
python -u "$REPO/scripts/run_phase1.py" --config "$CFG" --device cpu
echo "done $(date)"
