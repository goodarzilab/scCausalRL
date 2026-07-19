#!/bin/bash
#SBATCH --job-name=grn_legacy
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --output=logs/grn_legacy_%j.out
#SBATCH --error=logs/grn_legacy_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments_rl
export OMP_NUM_THREADS=12
echo "start $(date) on $(hostname)"
python -u analysis/grn_legacy_sweep.py
echo "done $(date)"
