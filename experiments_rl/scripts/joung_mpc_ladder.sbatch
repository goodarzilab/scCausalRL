#!/bin/bash
#SBATCH --job-name=jrl_mpc
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=110G
#SBATCH --cpus-per-task=12
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=4:00:00
#SBATCH --output=logs/joung_mpc_%j.out
#SBATCH --error=logs/joung_mpc_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments_rl
EXP19=/home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose
CKPT=$EXP19/results/grid/v2_esm_con_ugate/checkpoints/phase1_epoch60.pt
echo "start $(date) on $(hostname) | WM=ugate ep60 (gate-aware)"
# Joung value-add ladder + MPC on held-out combos. ALSO the stack smoke test:
# builds WMDataJoung (densifies the 45GB singles atlas), exercises the gate-aware
# do-operator, and scores baselines vs MPC on the 12 held-out combo pairs.
python -u run_fullpool_ladder.py \
    --dataset joung --wm_root "$EXP19" --checkpoint "$CKPT" --pool screened \
    --beam 16 --n_dag 8 --alpha 1.0 \
    --output rl0_mpc/results/joung_ugate
echo "done $(date)"
