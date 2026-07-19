#!/bin/bash
#SBATCH --job-name=jrl_distill
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=110G
#SBATCH --cpus-per-task=12
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=16:00:00
#SBATCH --output=logs/joung_distill_%j.out
#SBATCH --error=logs/joung_distill_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments_rl
EXP19=/home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose
CKPT=$EXP19/results/grid/v2_esm_con_ugate/checkpoints/phase1_epoch60.pt
echo "start $(date) on $(hostname)"
python -u rl3_mpc_distill/distill.py \
    --dataset joung --wm_root "$EXP19" --checkpoint "$CKPT" --pool screened \
    --k_max 2 --n_dag 8 --alpha 1.0 --mpc_beam 16 --bc_epochs 300 --ppo_iters 60000 \
    --wandb --run_name joung_distill_ugate --output rl3_mpc_distill/results/joung_ugate
echo "done $(date)"
