#!/bin/bash
#SBATCH --job-name=jrl_pposmoke
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=110G
#SBATCH --cpus-per-task=12
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=3:00:00
#SBATCH --output=logs/ppo_smoke_%j.out
#SBATCH --error=logs/ppo_smoke_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments_rl
EXP19=/home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose
CKPT=$EXP19/results/grid/v2_esm_con_ugate/checkpoints/phase1_epoch60.pt
COMMON="--dataset joung --wm_root $EXP19 --checkpoint $CKPT --pool screened --iters 6000 --n_dag 8"
run () { echo; echo "########## $1 ##########"; shift; "$@" 2>&1 | grep -iE "iter |eval|=====|match|entropy" | tail -25; }

# A) set-PPO RAW (replicate collapse), B) set-PPO RESID (fix), C) set-PPO RAW ent=0 (entropy-domination test)
run "A_set_raw"       python -u rl2_set_ppo/train.py $COMMON --reward_mode raw   --run_name smk_set_raw   --output rl2_set_ppo/results/smk_raw
run "B_set_resid"     python -u rl2_set_ppo/train.py $COMMON --reward_mode resid --run_name smk_set_resid --output rl2_set_ppo/results/smk_resid
run "C_set_raw_ent0"  python -u rl2_set_ppo/train.py $COMMON --reward_mode raw --entropy_coeff 0.0 --run_name smk_set_ent0 --output rl2_set_ppo/results/smk_ent0
# D) seq-PPO RESID (fix on the sequential variant)
run "D_seq_resid"     python -u rl1_seq_ppo/train.py $COMMON --reward_mode resid --run_name smk_seq_resid --output rl1_seq_ppo/results/smk_resid
echo; echo "done $(date)"
