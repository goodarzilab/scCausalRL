#!/bin/bash
#SBATCH --job-name=jrl_fixed
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=110G
#SBATCH --cpus-per-task=12
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=16:00:00
#SBATCH --output=logs/rlfixed_%x_%j.out
#SBATCH --error=logs/rlfixed_%x_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments_rl
EXP19=/home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose
CKPT="${CKPT:-$EXP19/results/grid/v2_esm_con_ugate/checkpoints/phase1_epoch60.pt}"
METHOD="${METHOD:?seq|set|distill}"; POOL="${POOL:-curated}"; REWARD="${REWARD:-blend}"; NAME="${NAME:?}"; ITERS="${ITERS:-80000}"
EXTRA="${EXTRA:-}"   # e.g. "--entropy_final 0.002 --lr_actor 1e-4"
COMMON="--dataset joung --wm_root $EXP19 --checkpoint $CKPT --pool $POOL --n_dag 8 --alpha 1.0 --curated_topk 256 --wandb --run_name $NAME"
echo "start $(date) on $(hostname) | $NAME method=$METHOD pool=$POOL reward=$REWARD iters=$ITERS extra=$EXTRA"
case "$METHOD" in
  seq)     python -u rl1_seq_ppo/train.py  $COMMON --reward_mode $REWARD --iters $ITERS --max_steps 2 $EXTRA --output rl1_seq_ppo/results/$NAME ;;
  set)     python -u rl2_set_ppo/train.py  $COMMON --reward_mode $REWARD --iters $ITERS --k_max 2  $EXTRA --output rl2_set_ppo/results/$NAME ;;
  distill) python -u rl3_mpc_distill/distill.py $COMMON --k_max 2 --mpc_beam 16 --bc_epochs 300 --ppo_iters 60000 --output rl3_mpc_distill/results/$NAME ;;
esac
echo "done $(date)"
