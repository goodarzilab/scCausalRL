#!/bin/bash
#SBATCH --job-name=fate_setppo
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=110G
#SBATCH --cpus-per-task=12
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=03:00:00
#SBATCH --output=logs/fate_setppo_%j.out
#SBATCH --error=logs/fate_setppo_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments_rl
EXP19=/home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose
CKPT=$EXP19/results/grid/v2_esm_con_ugate/checkpoints/phase1_epoch60.pt
echo "start $(date) on $(hostname)"
python -u analysis/fate_set_ppo.py --dataset joung --wm_root "$EXP19" --checkpoint "$CKPT" \
    --reward_mode resid --pool curated --iters 15000 --wandb --run_name joung_fate_set_ppo
echo "done $(date)"
