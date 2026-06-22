#!/bin/bash
#SBATCH --job-name=joung_heldeval
#SBATCH --partition=cpu_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --output=logs/heldeval_%j.out
#SBATCH --error=logs/heldeval_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas
mkdir -p logs results/heldout_eval
CKPT="${CKPT:-results/grid/exp17_joung_tfatlas/checkpoints/phase1_epoch80.pt}"
TAG="${TAG:-epoch80}"
echo "start $(date) on $(hostname) | ckpt=$CKPT tag=$TAG"
OMP_NUM_THREADS=8 python -u evaluation/heldout_zeroshot_eval.py --checkpoint "$CKPT" --tag "$TAG"
echo "done $(date)"
ls -la results/heldout_eval/$TAG/ 2>/dev/null
