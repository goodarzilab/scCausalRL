#!/bin/bash
#SBATCH --job-name=joung_shiftdiag
#SBATCH --partition=cpu_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=2:00:00
#SBATCH --output=logs/shiftdiag_%j.out
#SBATCH --error=logs/shiftdiag_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
CKPT="${CKPT:-results/grid/exp17_joung_tfatlas/checkpoints/phase1_epoch80.pt}"
TAG="${TAG:-epoch80}"
echo "start $(date) on $(hostname) | ckpt=$CKPT tag=$TAG"
OMP_NUM_THREADS=8 python -u evaluation/shift_diagnostics.py --checkpoint "$CKPT" --tag "$TAG" --device cpu
echo "done $(date)"
