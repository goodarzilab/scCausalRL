#!/bin/bash
#SBATCH --job-name=x20_bcft
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=8:00:00
#SBATCH --output=logs/bcft_%j.out
#SBATCH --error=logs/bcft_%j.err
set -uo pipefail
RL=/home/sgoudarzi/scCausalRL/experiments_rl
cd "$RL"; export PYTHONPATH="$RL" OMP_NUM_THREADS=8
python -u analysis/briggs_setppo_bcfinetune.py --kmax 7 --bc_epochs 500 --ft_iters 3000 --n_ctrl 256 --n_dag 6 2>&1 | grep -vE "Warning|warn|verbose|FutureWarning"
echo "fin $(date)"
