#!/bin/bash
#SBATCH --job-name=joung_epistasis_q
#SBATCH --partition=cpu_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=2:00:00
#SBATCH --output=logs/epistasis_q_%j.out
#SBATCH --error=logs/epistasis_q_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose
mkdir -p logs results/epistasis_quartile
TAG="${TAG:-epistasis_q}"
echo "start $(date) on $(hostname) | tag=$TAG"
OMP_NUM_THREADS=8 python -u evaluation/epistasis_quartile_eval.py --tag "$TAG"
echo "done $(date)"; cat results/epistasis_quartile/$TAG/summary.json 2>/dev/null | head -60
