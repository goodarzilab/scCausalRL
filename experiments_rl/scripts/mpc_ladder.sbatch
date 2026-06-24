#!/bin/bash
#SBATCH --job-name=rl_mpc_ladder
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=3:00:00
#SBATCH --output=logs/mpc_ladder_%j.out
#SBATCH --error=logs/mpc_ladder_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments_rl
echo "start $(date) on $(hostname)"
python -u run_fullpool_ladder.py --pool full --beam 16 --n_dag 8 \
    --output rl0_mpc/results/fullpool
echo "done $(date)"
