#!/bin/bash
#SBATCH --job-name=qc_joung
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=logs/qc_joung_%j.out
#SBATCH --error=logs/qc_joung_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments_rl
export OMP_NUM_THREADS=8 PYTHONPATH="$PWD:/home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose"
echo "start $(date) on $(hostname)"
python -u analysis/qc_joung.py
echo "done $(date)"
