#!/bin/bash
#SBATCH --job-name=rl_seq_ppo
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=10:00:00
#SBATCH --output=logs/seq_ppo_%j.out
#SBATCH --error=logs/seq_ppo_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments_rl
echo "start $(date) on $(hostname)"
python -u rl1_seq_ppo/train.py --pool full --iters 80000 --n_dag 8 --alpha 1.0 \
    --wandb --run_name seq_ppo_fullpool --output rl1_seq_ppo/results
echo "done $(date)"
