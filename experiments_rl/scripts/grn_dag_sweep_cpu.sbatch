#!/bin/bash
#SBATCH --job-name=grn_eval_cpu
#SBATCH --partition=cpu_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/grn_eval_cpu_%j.out
#SBATCH --error=logs/grn_eval_cpu_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments_rl
export OMP_NUM_THREADS=16
echo "start $(date) on $(hostname)"
python -u analysis/grn_dag_sweep.py --grid "${GRID:-grn}" --device cpu
echo "done $(date)"
