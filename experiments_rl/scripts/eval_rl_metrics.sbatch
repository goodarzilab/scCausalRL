#!/bin/bash
#SBATCH --job-name=jrl_fullmet
#SBATCH --partition=cpu_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=1:30:00
#SBATCH --output=logs/rl_fullmet_%j.out
#SBATCH --error=logs/rl_fullmet_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments_rl
echo "start $(date) on $(hostname)"
OMP_NUM_THREADS=8 python -u eval_rl_full_metrics.py --device cpu --output rl0_mpc/results/rl_full_metrics
echo "done $(date)"
