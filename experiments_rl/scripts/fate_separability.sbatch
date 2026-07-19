#!/bin/bash
#SBATCH --job-name=fate_sep
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=110G
#SBATCH --cpus-per-task=12
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=01:00:00
#SBATCH --output=logs/fate_sep_%j.out
#SBATCH --error=logs/fate_sep_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments_rl
EXP19=/home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose
TAG="${TAG:-v2_esm_con_ugate}"
CKPT="${CKPT:-$EXP19/results/grid/$TAG/checkpoints/phase1_epoch60.pt}"
echo "start $(date) on $(hostname)  TAG=$TAG  CKPT=$CKPT"
python -u analysis/fate_separability.py \
    --dataset joung --wm_root "$EXP19" --checkpoint "$CKPT" --qmax 0.10
echo "done $(date)"
