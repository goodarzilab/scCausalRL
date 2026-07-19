#!/bin/bash
#SBATCH --job-name=jrl_transfer
#SBATCH --partition=cpu_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=2:00:00
#SBATCH --output=logs/transfer_norman_%j.out
#SBATCH --error=logs/transfer_norman_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments_rl
echo "start $(date) on $(hostname)"
OMP_NUM_THREADS=8 python -u transfer_joung_to_norman.py \
    --device cpu --output rl0_mpc/results/transfer_norman
echo "done $(date)"; cat rl0_mpc/results/transfer_norman/summary.json 2>/dev/null
