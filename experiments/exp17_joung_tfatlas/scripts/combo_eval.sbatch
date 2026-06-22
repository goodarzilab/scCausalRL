#!/bin/bash
#SBATCH --job-name=joung_comboeval
#SBATCH --partition=cpu_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=2:00:00
#SBATCH --output=logs/comboeval_%j.out
#SBATCH --error=logs/comboeval_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas
mkdir -p logs results/combo_eval
CKPT="${CKPT:-results/grid/exp17_joung_tfatlas/checkpoints/phase1_epoch80.pt}"
TAG="${TAG:-epoch80}"
echo "start $(date) on $(hostname) | ckpt=$CKPT tag=$TAG"
OMP_NUM_THREADS=8 python -u evaluation/combo_triple_eval.py --checkpoint "$CKPT" --tag "$TAG"
echo "done $(date)"; cat results/combo_eval/$TAG/summary.json 2>/dev/null
