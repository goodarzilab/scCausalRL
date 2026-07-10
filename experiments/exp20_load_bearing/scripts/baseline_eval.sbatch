#!/bin/bash
#SBATCH --job-name=joung_baseline
#SBATCH --partition=cpu_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=2:00:00
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose
mkdir -p logs results/baseline_eval
TAG="${TAG:-baselines}"
echo "start $(date) on $(hostname) | tag=$TAG"
OMP_NUM_THREADS=8 python -u evaluation/baseline_eval.py --tag "$TAG"
echo "done $(date)"; cat results/baseline_eval/$TAG/summary.json 2>/dev/null | head -50
