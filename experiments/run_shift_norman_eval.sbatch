#!/bin/bash
#SBATCH --job-name=cbrl_norman_eval
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=240G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=03:00:00
#SBATCH --output=/home/sgoudarzi/scCausalRL/experiments/logs/%x_%j.out
#SBATCH --error=/home/sgoudarzi/scCausalRL/experiments/logs/%x_%j.err
set -euo pipefail
EXP_DIR="${EXP_DIR:?}"; CKPT="${CKPT:?}"; CFG="${CFG:?}"; OUTDIR="${OUTDIR:?}"
CMODE="${CMODE:-both}"; WANDB_FLAG="${WANDB_FLAG:-}"
case "$CFG" in /*) CFG_ABS="$CFG";; *) CFG_ABS="$EXP_DIR/$CFG";; esac
mkdir -p "$OUTDIR"
echo "=== external Norman eval === job ${SLURM_JOB_ID:-<i>} $(hostname) $(date)"
echo "Ckpt: $CKPT | Out: $OUTDIR | c_mode: $CMODE"
cd "$EXP_DIR"
python -u "$EXP_DIR/evaluation/run_shift_eval_norman.py" \
    --checkpoint "$CKPT" --config "$CFG_ABS" --output_dir "$OUTDIR" \
    --c_mode "$CMODE" --device cuda $WANDB_FLAG
echo "=== done $(date) === summary: $OUTDIR/summary.csv"
