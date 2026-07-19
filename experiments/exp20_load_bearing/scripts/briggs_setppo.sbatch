#!/bin/bash
#SBATCH --job-name=x20_setppo
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=logs/setppo_%j.out
#SBATCH --error=logs/setppo_%j.err
set -uo pipefail
RL=/home/sgoudarzi/scCausalRL/experiments_rl
cd "$RL"; export PYTHONPATH="$RL" OMP_NUM_THREADS=8
python -u analysis/briggs_fate_set_ppo.py --iters 6000 --n_ctrl 256 --n_dag 6 \
  --lam 1 --support_gamma 2 --kmax 7 2>&1 | grep -vE "Warning|warn|verbose|FutureWarning"
echo "fin $(date)"
