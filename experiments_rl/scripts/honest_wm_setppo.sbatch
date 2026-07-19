#!/bin/bash
#SBATCH --job-name=honestWM_setppo
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=110G
#SBATCH --cpus-per-task=12
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=04:00:00
#SBATCH --output=logs/honestWM_setppo_%j.out
#SBATCH --error=logs/honestWM_setppo_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments_rl
EXP19=/home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose
CKPT=$EXP19/results/grid/v2_esm_con_ugate_combo_ho12/checkpoints/phase1_epoch60.pt
echo "start $(date) on $(hostname) | HONEST combo WM (12 RL-test pairs held out of WM training)"
python -u rl2_set_ppo/train.py --dataset joung --wm_root "$EXP19" --checkpoint "$CKPT" \
    --reward_mode resid --pool curated --iters 80000 --k_max 2 --n_dag 8 --entropy_final 0.002 \
    --output rl2_set_ppo/results/joung_set_resid_curated_honestComboWM \
    --wandb --run_name joung_set_resid_curated_honestComboWM
echo "done $(date)"
