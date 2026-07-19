#!/bin/bash
#SBATCH --job-name=x20_setppodbg
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#SBATCH --output=logs/setppodbg_%j.out
#SBATCH --error=logs/setppodbg_%j.err
set -uo pipefail
RL=/home/sgoudarzi/scCausalRL/experiments_rl
cd "$RL"; export PYTHONPATH="$RL" OMP_NUM_THREADS=8
python -u analysis/briggs_setppo_debug.py --iters 2500 --n_ctrl 128 --n_dag 4 --update_every 128 2>&1 | grep -vE "Warning|warn|verbose|FutureWarning"
echo "fin $(date)"
