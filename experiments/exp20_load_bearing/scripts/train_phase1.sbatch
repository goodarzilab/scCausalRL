#!/bin/bash
#SBATCH --job-name=exp17_joung_train
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=300G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
set -euo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose
OUT=$REPO/results/grid/exp17_joung_tfatlas
mkdir -p "$OUT" "$REPO/logs"
cd "$OUT"                                   # checkpoints/ + logs/ land here
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
echo "start $(date) on $(hostname)"
python -u "$REPO/scripts/run_phase1.py" --config "$REPO/configs/joung.yaml" --device cuda
echo "done $(date)"
ls -la checkpoints/ | tail -5
