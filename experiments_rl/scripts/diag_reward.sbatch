#!/bin/bash
#SBATCH --job-name=jrl_diag
#SBATCH --partition=cpu_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=1:30:00
#SBATCH --output=logs/diag_reward_%j.out
#SBATCH --error=logs/diag_reward_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments_rl
EXP19=/home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose
CKPT=$EXP19/results/grid/v2_esm_con_ugate/checkpoints/phase1_epoch60.pt
echo "start $(date) on $(hostname)"
OMP_NUM_THREADS=8 python -u diag_reward_landscape.py \
    --dataset joung --wm_root "$EXP19" --checkpoint "$CKPT" --device cpu \
    --output rl0_mpc/results/diag
echo "done $(date)"
