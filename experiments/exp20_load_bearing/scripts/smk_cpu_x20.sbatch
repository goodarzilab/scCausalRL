#!/bin/bash
#SBATCH --job-name=x20smk
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=340G
#SBATCH --time=2:00:00
#SBATCH --output=logs/x20smk_%x_%j.out
#SBATCH --error=logs/x20smk_%x_%j.err
set -euo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
CFG="${CFG:?set CFG}"; TAG="${TAG:?set TAG}"
case "$CFG" in /*) ;; *) CFG="$REPO/$CFG" ;; esac
OUT="$REPO/results/grid/$TAG"; mkdir -p "$OUT" "$REPO/logs"; cd "$OUT"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"; export OMP_NUM_THREADS=16
echo "start $(date) on $(hostname) | CPU SMOKE | cfg=$CFG tag=$TAG"
python -u "$REPO/scripts/run_phase1.py" --config "$CFG" --device cpu
echo "done $(date)"; ls -la checkpoints/ 2>/dev/null | tail -4