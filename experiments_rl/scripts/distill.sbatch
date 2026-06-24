#!/bin/bash
#SBATCH --job-name=rl_distill
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=10:00:00
#SBATCH --output=logs/distill_%j.out
#SBATCH --error=logs/distill_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments_rl
echo "start $(date) on $(hostname)"
python -u rl3_mpc_distill/distill.py --pool full --k_max 2 --n_dag 8 --alpha 1.0 \
    --mpc_beam 16 --bc_epochs 300 --ppo_iters 60000 \
    --wandb --run_name set_ppo_distill_fullpool --output rl3_mpc_distill/results
echo "done $(date)"
