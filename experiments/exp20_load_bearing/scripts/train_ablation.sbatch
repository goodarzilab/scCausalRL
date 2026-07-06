#!/bin/bash
#SBATCH --job-name=exp20_abl
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=340G
#SBATCH --cpus-per-task=12
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=16:00:00   # runs take ~11h; short walltime = backfill-eligible (2-day blocked on reservations)
#SBATCH --output=logs/abl_%x_%j.out
#SBATCH --error=logs/abl_%x_%j.err
set -euo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
CFG="${CFG:?set CFG}"; TAG="${TAG:?set TAG}"
case "$CFG" in /*) ;; *) CFG="$REPO/$CFG" ;; esac
OUT="$REPO/results/grid/$TAG"; mkdir -p "$OUT" "$REPO/logs"; cd "$OUT"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
echo "start $(date) on $(hostname) | cfg=$CFG tag=$TAG"
python -u "$REPO/scripts/run_phase1.py" --config "$CFG" --device cuda
echo "done $(date)"
